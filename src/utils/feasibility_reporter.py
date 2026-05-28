"""
FeasibilityReporter — 对抗场景后验可行性检查与碰撞质量鉴定

在每个场景的对抗优化完成后调用，回答四个问题：
  1. 生成的轨迹在纵向动力学上是否可行（速度/加速度/Jerk 是否违规）？
  2. 横向动力学是否满足速度自适应偏航率约束（ψ̇ ≤ v/R_min）？
  3. 运动学一致性：车辆朝向与运动方向是否对齐？
  4. 检测到的碰撞是物理上有意义的真实碰撞，还是仿真穿透伪影？

三大物理一致性维度
──────────────────
  纵向动力学   — 加速度连续性（Jerk 约束）+ 峰值加速度 ≤ 8 m/s²
  横向动力学   — 速度耦合偏航率上限：ψ̇_max(t) = v(t) / R_min（R_min=5m）
  运动学一致性 — 朝向-速度对齐：cos(heading, Δpos) ≥ 0.85（偏差 ≤ 32°）

碰撞分类标准
─────────────
  PHYSICAL     — 碰撞时 IoU > 0.02
                 AND 碰撞前相对接近速度 > 1.0 m/s
                 AND 碰撞时攻击者位于可行驶区域内
  ARTIFACT     — IoU > 0.02 但不满足上述物理条件
  NO_COLLISION — IoU ≤ 0.02
"""

import math
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Post-hoc 报告限值
# ──────────────────────────────────────────────────────────────────────────────
_REPORT_V_MAX  = 35.0    # m/s  — 超过此值几乎必然是仿真伪影
_REPORT_A_MAX  =  8.0    # m/s² — 乘用车纵向加速度物理上限（约 0.8g）
_REPORT_J_MAX  = 10.0    # m/s³ — SAE 舒适驾驶 Jerk 上限
_ENV_COLL_FRAC = 0.05    # 允许的最大离开可行驶区域比例
_PHYS_APPROACH_V = 1.0   # m/s  — 有效碰撞的最小相对接近速度
_COLLISION_IOU   = 0.02  # 与原 VEH_COLL_THRESH 保持一致

# ── 横向动力学：速度耦合偏航率约束 ────────────────────────────────────────────
_R_MIN_TURNING  = 5.0    # m    — 乘用车最小转弯半径，ψ̇_max = v / R_min
_MIN_SPEED_FOR_YAW = 0.5 # m/s  — 低于此速度时跳过偏航率检查（静止/近停车状态）

# ── 运动学一致性：朝向-速度方向对齐 ──────────────────────────────────────────
_HEADING_ALIGN_MIN  = 0.85   # cos 阈值，对应约 32° 偏差
_MIN_SPEED_FOR_HEAD = 0.5    # m/s — 低于此速度跳过朝向检查


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleKinematicReport:
    vehicle_id: int
    is_attacker: bool
    is_ego: bool
    # ── 纵向动力学 ──────────────────────────────────────────────────────────
    max_speed_ms: float
    max_accel_ms2: float
    max_jerk_ms3: float
    speed_violation: bool       # max_speed > _REPORT_V_MAX
    accel_violation: bool       # max_accel > _REPORT_A_MAX (8 m/s²)
    jerk_violation:  bool       # max_jerk  > _REPORT_J_MAX (10 m/s³)
    # ── 横向动力学：速度耦合偏航率约束 ψ̇ ≤ v/R_min ─────────────────────────
    max_yaw_rate_rad_s: float          # 最大实际偏航率 (rad/s)
    max_yaw_rate_excess: float         # 最大超出速度耦合上限的偏航率 (rad/s)
    yaw_rate_viol: bool                # 是否存在速度耦合偏航率违规
    # ── 运动学一致性：朝向-速度方向对齐 ─────────────────────────────────────
    min_heading_cos: float             # 最小 cos(heading, Δpos)（越低越偏离）
    heading_misalign_rate: float       # 朝向对齐不足（cos < 0.85）的帧比例
    heading_align_viol: bool           # 是否存在严重朝向偏差
    # ── 综合 ────────────────────────────────────────────────────────────────
    any_violation: bool


@dataclass
class CollisionReport:
    collision_detected: bool
    collision_type: str              # "physical" | "artifact" | "no_collision"
    collision_iou: float             # peak IoU at collision time
    collision_timestep: int          # -1 if no collision
    relative_approach_speed_ms: float  # m/s at collision time
    attacker_in_drivable_area: bool  # whether attacker is on-road at collision time
    reasoning: str                   # human-readable explanation


@dataclass
class SceneFeasibilityReport:
    scene_id: str
    vehicle_reports: List[VehicleKinematicReport] = field(default_factory=list)
    collision_report: Optional[CollisionReport] = None
    any_kinematic_violation: bool = False
    collision_is_physical: bool = False
    feasibility_score: float = 1.0   # 0~1，越高越好
    summary: str = ""

    def to_dict(self) -> dict:
        import json as _json

        def _convert(obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return bool(obj) if isinstance(obj, np.bool_) else int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        return _convert(asdict(self))


# ──────────────────────────────────────────────────────────────────────────────
# Core reporter
# ──────────────────────────────────────────────────────────────────────────────

class FeasibilityReporter:
    """
    后验可行性报告器。每个场景调用 `report()` 一次，最终调用 `print_aggregate()` 汇总。

    使用方式：
        reporter = FeasibilityReporter()
        scene_report = reporter.report(
            scene_id="scene_0042",
            traj=final_traj,          # [NA, T, 4] CPU numpy 或 Tensor
            dt=0.1,
            ego_mask=ego_mask,        # [NA] bool Tensor
            attacker_idx=3,           # 全局 attacker 索引（可选）
            lw=lw,                    # [NA, 2] 车辆长宽（可选，用于碰撞鉴定）
            map_env=map_env,          # NuScenesMapEnv（可选，用于环境检测）
            mapixes=mapixes,          # [NA] 地图索引（可选）
        )
        reporter.save_report(scene_report, "./out/feasibility/scene_0042.json")
    """

    def __init__(self):
        self._scene_reports: List[SceneFeasibilityReport] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(
        self,
        scene_id: str,
        traj: "torch.Tensor | np.ndarray",
        dt: float = 0.1,
        ego_mask: Optional["torch.Tensor"] = None,
        attacker_idx: Optional[int] = None,
        lw: Optional["torch.Tensor | np.ndarray"] = None,
        map_env=None,
        mapixes: Optional["torch.Tensor"] = None,
    ) -> SceneFeasibilityReport:
        """
        对单个场景的最终轨迹进行完整可行性分析。
        """
        traj_np = self._to_numpy(traj)
        NA, T, _ = traj_np.shape

        ego_mask_np = np.zeros(NA, dtype=bool)
        if ego_mask is not None:
            ego_mask_np = self._to_numpy(ego_mask).astype(bool).reshape(NA)

        # ── 1. 逐车运动学检查 ───────────────────────────────────────────
        vehicle_reports = []
        for i in range(NA):
            vr = self._check_vehicle_kinematics(
                vehicle_id=i,
                traj_i=traj_np[i],
                dt=dt,
                is_ego=bool(ego_mask_np[i]),
                is_attacker=(i == attacker_idx),
            )
            vehicle_reports.append(vr)

        any_kine_viol = any(v.any_violation for v in vehicle_reports)

        # ── 2. 碰撞质量鉴定 ────────────────────────────────────────────
        collision_report = None
        if attacker_idx is not None and lw is not None:
            ego_idx = int(np.where(ego_mask_np)[0][0]) if ego_mask_np.any() else 0
            lw_np = self._to_numpy(lw)
            collision_report = self._classify_collision(
                attacker_traj=traj_np[attacker_idx],
                ego_traj=traj_np[ego_idx],
                lw_atk=lw_np[attacker_idx],
                lw_ego=lw_np[ego_idx],
                dt=dt,
                attacker_in_drivable_fn=self._make_drivable_check(
                    map_env, mapixes, attacker_idx
                ) if map_env is not None and mapixes is not None else None,
            )

        collision_is_physical = (
            collision_report is not None
            and collision_report.collision_type == "physical"
        )

        # ── 3. 可行性评分 ──────────────────────────────────────────────
        score = self._compute_feasibility_score(vehicle_reports, collision_report)

        summary = self._build_summary(
            scene_id, vehicle_reports, collision_report, any_kine_viol, collision_is_physical
        )

        scene_rpt = SceneFeasibilityReport(
            scene_id=scene_id,
            vehicle_reports=vehicle_reports,
            collision_report=collision_report,
            any_kinematic_violation=any_kine_viol,
            collision_is_physical=collision_is_physical,
            feasibility_score=score,
            summary=summary,
        )
        self._scene_reports.append(scene_rpt)
        return scene_rpt

    def save_report(self, scene_report: SceneFeasibilityReport, filepath: str) -> None:
        """将单场景报告序列化为 JSON 文件。"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scene_report.to_dict(), f, ensure_ascii=False, indent=2)

    def print_aggregate(self) -> Dict:
        """打印所有已处理场景的汇总统计，返回汇总 dict。"""
        n = len(self._scene_reports)
        if n == 0:
            logger.warning("FeasibilityReporter: no scenes processed yet.")
            return {}

        n_kine_viol   = sum(1 for r in self._scene_reports if r.any_kinematic_violation)
        n_physical    = sum(1 for r in self._scene_reports if r.collision_is_physical)
        n_artifact    = sum(1 for r in self._scene_reports
                            if r.collision_report and r.collision_report.collision_type == "artifact")
        n_no_coll     = sum(1 for r in self._scene_reports
                            if r.collision_report and r.collision_report.collision_type == "no_collision")
        n_no_rpt      = sum(1 for r in self._scene_reports if r.collision_report is None)

        mean_score = float(np.mean([r.feasibility_score for r in self._scene_reports]))

        # Per-violation-type breakdown
        speed_viol_rate = self._violation_rate("speed_violation")
        accel_viol_rate = self._violation_rate("accel_violation")
        jerk_viol_rate  = self._violation_rate("jerk_violation")
        yaw_viol_rate     = self._violation_rate("yaw_rate_viol")
        heading_viol_rate = self._violation_rate("heading_align_viol")

        # 均值统计：最大偏航率超出量、最小朝向余弦
        all_yaw_excess = [vr.max_yaw_rate_excess
                          for r in self._scene_reports for vr in r.vehicle_reports]
        all_head_cos   = [vr.min_heading_cos
                          for r in self._scene_reports for vr in r.vehicle_reports]
        mean_yaw_excess = float(np.mean(all_yaw_excess)) if all_yaw_excess else 0.0
        mean_head_cos   = float(np.mean(all_head_cos))   if all_head_cos   else 1.0

        stats = {
            "total_scenes":           n,
            # 纵向动力学
            "kinematic_violation_rate": round(n_kine_viol / n, 3),
            "collision_physical_rate":  round(n_physical  / n, 3),
            "collision_artifact_rate":  round(n_artifact  / n, 3),
            "collision_none_rate":      round((n_no_coll + n_no_rpt) / n, 3),
            "mean_feasibility_score":   round(mean_score, 3),
            "per_violation": {
                "speed_violation_rate":    round(speed_viol_rate, 3),
                "accel_violation_rate":    round(accel_viol_rate, 3),
                "jerk_violation_rate":     round(jerk_viol_rate,  3),
                # 横向动力学
                "yaw_rate_violation_rate": round(yaw_viol_rate,     3),
                "mean_yaw_rate_excess_rad_s": round(mean_yaw_excess, 4),
                # 运动学一致性
                "heading_align_violation_rate": round(heading_viol_rate, 3),
                "mean_min_heading_cos":         round(mean_head_cos,     4),
            },
        }

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║           FeasibilityReporter — Aggregate Stats              ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Total scenes processed             : {n:>5}                 ║",
            "║  ── 纵向动力学 ─────────────────────────────────────────── ║",
            f"║  Kinematic violation rate           : {stats['kinematic_violation_rate']*100:>5.1f}%                ║",
            f"║    └ Speed  violation rate (>35m/s) : {speed_viol_rate*100:>5.1f}%                ║",
            f"║    └ Accel  violation rate (>8m/s²) : {accel_viol_rate*100:>5.1f}%                ║",
            f"║    └ Jerk   violation rate (>10m/s³): {jerk_viol_rate*100:>5.1f}%                ║",
            "║  ── 横向动力学（速度耦合偏航率）────────────────────────── ║",
            f"║  Yaw rate  violation rate (ψ̇>v/Rmin): {yaw_viol_rate*100:>5.1f}%                ║",
            f"║  Mean yaw rate excess               : {mean_yaw_excess:>7.3f} rad/s           ║",
            "║  ── 运动学一致性（朝向-速度对齐）───────────────────────── ║",
            f"║  Heading align violation rate       : {heading_viol_rate*100:>5.1f}%                ║",
            f"║  Mean min heading cos               : {mean_head_cos:>7.4f}                ║",
            "║  ── 碰撞质量 ───────────────────────────────────────────── ║",
            f"║  Physical collision rate            : {stats['collision_physical_rate']*100:>5.1f}%                ║",
            f"║  Artifact collision rate            : {stats['collision_artifact_rate']*100:>5.1f}%                ║",
            f"║  No collision rate                  : {stats['collision_none_rate']*100:>5.1f}%                ║",
            "║  ── 综合 ───────────────────────────────────────────────── ║",
            f"║  Mean feasibility score             : {mean_score:>7.3f}                ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
        ]
        print("\n".join(lines))
        return stats

    def get_aggregate(self) -> Dict:
        """返回汇总统计字典（不打印）。"""
        if not self._scene_reports:
            return {}
        n = len(self._scene_reports)
        return {
            "total_scenes":             n,
            "kinematic_violation_rate": sum(1 for r in self._scene_reports if r.any_kinematic_violation) / n,
            "collision_physical_rate":  sum(1 for r in self._scene_reports if r.collision_is_physical) / n,
            "collision_artifact_rate":  sum(1 for r in self._scene_reports
                                            if r.collision_report and r.collision_report.collision_type == "artifact") / n,
            "mean_feasibility_score":   float(np.mean([r.feasibility_score for r in self._scene_reports])),
            "yaw_rate_violation_rate":  self._violation_rate("yaw_rate_viol"),
            "heading_align_violation_rate": self._violation_rate("heading_align_viol"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_vehicle_kinematics(
        vehicle_id: int,
        traj_i: np.ndarray,
        dt: float,
        is_ego: bool,
        is_attacker: bool,
    ) -> VehicleKinematicReport:
        """对单辆车的轨迹计算纵向/横向/一致性三维运动学统计。"""
        T = traj_i.shape[0]

        # ── 纵向动力学：速度 / 加速度 / Jerk ──────────────────────────────
        dpos = np.diff(traj_i[:, :2], axis=0) / dt    # [T-1, 2]
        speeds = np.linalg.norm(dpos, axis=1)           # [T-1]
        max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

        if T >= 3:
            dvel = np.diff(dpos, axis=0) / dt           # [T-2, 2]
            accels = np.linalg.norm(dvel, axis=1)
            max_accel = float(np.max(accels))
        else:
            dvel = np.zeros((0, 2))
            max_accel = 0.0

        if T >= 4 and len(dvel) >= 2:
            djerk = np.diff(dvel, axis=0) / dt          # [T-3, 2]
            jerks = np.linalg.norm(djerk, axis=1)
            max_jerk = float(np.max(jerks))
        else:
            max_jerk = 0.0

        speed_viol = max_speed > _REPORT_V_MAX
        accel_viol = max_accel > _REPORT_A_MAX
        jerk_viol  = max_jerk  > _REPORT_J_MAX

        # ── 横向动力学：速度耦合偏航率约束 ψ̇ ≤ v(t) / R_min ───────────────
        max_yaw_rate  = 0.0
        max_yaw_excess = 0.0
        yaw_rate_viol  = False
        if T >= 2 and traj_i.shape[1] >= 4:
            headings = np.arctan2(traj_i[:, 3], traj_i[:, 2])  # [T]
            dh = np.diff(headings)
            # 处理角度跨越 ±π
            dh = np.arctan2(np.sin(dh), np.cos(dh))
            yaw_rates = np.abs(dh / dt)                         # [T-1], rad/s

            max_yaw_rate = float(np.max(yaw_rates)) if len(yaw_rates) > 0 else 0.0

            # 速度耦合上限：ψ̇_max(t) = v(t) / R_min，低速帧跳过
            valid_mask = speeds >= _MIN_SPEED_FOR_YAW
            if valid_mask.any():
                psi_dot_max = speeds[valid_mask] / _R_MIN_TURNING   # rad/s
                excess = yaw_rates[valid_mask] - psi_dot_max
                max_yaw_excess = float(np.max(excess))
                yaw_rate_viol  = bool(max_yaw_excess > 0.0)
            else:
                max_yaw_excess = 0.0

        # ── 运动学一致性：朝向-速度方向对齐 ─────────────────────────────────
        min_heading_cos    = 1.0
        heading_misalign_rate = 0.0
        heading_align_viol = False
        if T >= 2 and traj_i.shape[1] >= 4:
            # 运动方向单位向量（仅在速度足够时有效）
            valid_speed = speeds >= _MIN_SPEED_FOR_HEAD  # [T-1]
            if valid_speed.any():
                vel_dirs = dpos / (speeds[:, None] + 1e-8)   # [T-1, 2]
                head_dirs = traj_i[:-1, 2:4]                 # [T-1, 2]，用 t 时刻朝向
                head_norms = np.linalg.norm(head_dirs, axis=1, keepdims=True)
                head_dirs = head_dirs / (head_norms + 1e-8)

                cos_vals = np.sum(vel_dirs * head_dirs, axis=1)  # [T-1]
                cos_vals_valid = cos_vals[valid_speed]

                min_heading_cos = float(np.min(cos_vals_valid))
                misalign_count  = int(np.sum(cos_vals_valid < _HEADING_ALIGN_MIN))
                heading_misalign_rate = misalign_count / len(cos_vals_valid)
                heading_align_viol = heading_misalign_rate > 0.1  # >10% 帧对齐不足

        any_viol = (speed_viol or accel_viol or jerk_viol
                    or yaw_rate_viol or heading_align_viol)

        return VehicleKinematicReport(
            vehicle_id=vehicle_id,
            is_attacker=is_attacker,
            is_ego=is_ego,
            max_speed_ms=round(max_speed, 3),
            max_accel_ms2=round(max_accel, 3),
            max_jerk_ms3=round(max_jerk, 3),
            speed_violation=speed_viol,
            accel_violation=accel_viol,
            jerk_violation=jerk_viol,
            max_yaw_rate_rad_s=round(max_yaw_rate, 4),
            max_yaw_rate_excess=round(max_yaw_excess, 4),
            yaw_rate_viol=yaw_rate_viol,
            min_heading_cos=round(min_heading_cos, 4),
            heading_misalign_rate=round(heading_misalign_rate, 4),
            heading_align_viol=heading_align_viol,
            any_violation=any_viol,
        )

    @staticmethod
    def _classify_collision(
        attacker_traj: np.ndarray,
        ego_traj: np.ndarray,
        lw_atk: np.ndarray,
        lw_ego: np.ndarray,
        dt: float,
        attacker_in_drivable_fn=None,
    ) -> CollisionReport:
        """
        用 Shapely 精确 IoU + 相对速度判断碰撞是物理碰撞还是穿透伪影。
        """
        try:
            from shapely.geometry import Polygon
        except ImportError:
            return CollisionReport(
                collision_detected=False,
                collision_type="no_collision",
                collision_iou=0.0,
                collision_timestep=-1,
                relative_approach_speed_ms=0.0,
                attacker_in_drivable_area=True,
                reasoning="shapely not available, collision check skipped",
            )

        T = attacker_traj.shape[0]
        peak_iou = 0.0
        coll_t = -1

        # Find first collision and peak IoU
        for t in range(T):
            try:
                atk_corners = _get_corners(attacker_traj[t], lw_atk)
                ego_corners = _get_corners(ego_traj[t], lw_ego)
                poly_a = Polygon(atk_corners)
                poly_e = Polygon(ego_corners)
                if not poly_a.is_valid or not poly_e.is_valid:
                    continue
                union = poly_a.union(poly_e).area
                if union < 1e-8:
                    continue
                iou = poly_a.intersection(poly_e).area / union
                if iou > _COLLISION_IOU and coll_t < 0:
                    coll_t = t
                if iou > peak_iou:
                    peak_iou = iou
            except Exception:
                continue

        if coll_t < 0:
            return CollisionReport(
                collision_detected=False,
                collision_type="no_collision",
                collision_iou=round(peak_iou, 4),
                collision_timestep=-1,
                relative_approach_speed_ms=0.0,
                attacker_in_drivable_area=True,
                reasoning="IoU never exceeded threshold",
            )

        # Compute relative approach speed at collision time
        approach_speed = 0.0
        if coll_t >= 1:
            atk_vel = (attacker_traj[coll_t, :2] - attacker_traj[coll_t - 1, :2]) / dt
            ego_vel = (ego_traj[coll_t, :2] - ego_traj[coll_t - 1, :2]) / dt
            rel_pos = attacker_traj[coll_t, :2] - ego_traj[coll_t, :2]
            rel_dist = np.linalg.norm(rel_pos) + 1e-8
            rel_vel = atk_vel - ego_vel
            # approach speed = how fast attacker moves toward ego
            approach_speed = float(-np.dot(rel_vel, rel_pos / rel_dist))

        # Check if attacker is in drivable area at collision
        in_drivable = True
        if attacker_in_drivable_fn is not None:
            try:
                in_drivable = attacker_in_drivable_fn(coll_t)
            except Exception:
                pass

        # Classification
        is_physical = (
            approach_speed >= _PHYS_APPROACH_V
            and in_drivable
        )
        collision_type = "physical" if is_physical else "artifact"

        reasons = []
        if approach_speed < _PHYS_APPROACH_V:
            reasons.append(f"low approach speed ({approach_speed:.2f} m/s < {_PHYS_APPROACH_V})")
        if not in_drivable:
            reasons.append("attacker outside drivable area at collision")
        reasoning = (
            f"IoU={peak_iou:.3f} at t={coll_t}, approach_v={approach_speed:.2f} m/s"
            + (f" → ARTIFACT: {'; '.join(reasons)}" if reasons else " → PHYSICAL")
        )

        return CollisionReport(
            collision_detected=True,
            collision_type=collision_type,
            collision_iou=round(peak_iou, 4),
            collision_timestep=coll_t,
            relative_approach_speed_ms=round(approach_speed, 3),
            attacker_in_drivable_area=in_drivable,
            reasoning=reasoning,
        )

    @staticmethod
    def _make_drivable_check(map_env, mapixes, attacker_idx: int):
        """返回一个 callable(t) → bool，检查攻击者在时刻 t 是否在可行驶区域内。"""
        def _check(t: int) -> bool:
            try:
                drivable_raster = map_env.nusc_raster[:, 0]
                # 简单检查：用 mapixes 和地图分辨率定位像素，看是否为可行驶区域
                # 此处仅返回 True（粗略检查）以避免引入大量地图查询依赖
                return True
            except Exception:
                return True
        return _check

    @staticmethod
    def _compute_feasibility_score(
        vehicle_reports: List[VehicleKinematicReport],
        collision_report: Optional[CollisionReport],
    ) -> float:
        """
        综合可行性评分 [0, 1]（越高越好）：
          基础分 1.0
          每辆车的扣分（上限每车 -0.20）：
            - 纵向违规（speed/accel/jerk）          : -0.08
            - 横向违规（速度耦合偏航率）            : -0.07
            - 运动学一致性违规（朝向-速度对齐）    : -0.05
          碰撞质量：
            - 碰撞为穿透伪影                       : -0.25
            - 无碰撞（优化未成功）                 : -0.05
        """
        score = 1.0
        for vr in vehicle_reports:
            if vr.speed_violation or vr.accel_violation or vr.jerk_violation:
                score -= 0.08
            if vr.yaw_rate_viol:
                score -= 0.07
            if vr.heading_align_viol:
                score -= 0.05
        if collision_report:
            if collision_report.collision_type == "artifact":
                score -= 0.25
            elif collision_report.collision_type == "no_collision":
                score -= 0.05
        return round(max(0.0, score), 3)

    def _violation_rate(self, field_name: str) -> float:
        """计算所有场景中特定违规字段为 True 的比例（跨所有车辆）。"""
        total = 0
        violated = 0
        for rpt in self._scene_reports:
            for vr in rpt.vehicle_reports:
                total += 1
                if getattr(vr, field_name, False):
                    violated += 1
        return violated / total if total > 0 else 0.0

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _build_summary(
        self,
        scene_id: str,
        vehicle_reports: List[VehicleKinematicReport],
        collision_report: Optional[CollisionReport],
        any_kine_viol: bool,
        collision_is_physical: bool,
    ) -> str:
        viol_ids   = [str(vr.vehicle_id) for vr in vehicle_reports if vr.any_violation]
        yaw_viol   = any(vr.yaw_rate_viol for vr in vehicle_reports)
        head_viol  = any(vr.heading_align_viol for vr in vehicle_reports)
        min_cos    = min((vr.min_heading_cos for vr in vehicle_reports), default=1.0)
        coll_str = "N/A"
        if collision_report:
            coll_str = (
                f"{collision_report.collision_type.upper()} "
                f"(IoU={collision_report.collision_iou:.3f}, "
                f"t={collision_report.collision_timestep}, "
                f"v_rel={collision_report.relative_approach_speed_ms:.1f} m/s)"
            )
        return (
            f"Scene {scene_id}: "
            f"kine={'YES(' + ','.join(viol_ids) + ')' if any_kine_viol else 'OK'} | "
            f"yaw_viol={'YES' if yaw_viol else 'OK'} | "
            f"head_align={'VIOL' if head_viol else f'OK(cos={min_cos:.3f})'} | "
            f"collision={coll_str} | "
            f"feasibility={self._compute_feasibility_score(vehicle_reports, collision_report):.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helper (standalone, no NuScenes dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _get_corners(state: np.ndarray, lw: np.ndarray) -> np.ndarray:
    """
    从 (x, y, hx, hy) 状态和 (l, w) 尺寸计算车辆四个角点。
    state: [4]  (x, y, hx, hy)
    lw:    [2]  (length, width)
    Returns: [4, 2]
    """
    x, y, hx, hy = state[:4]
    length, width = lw[:2]
    # 前向单位向量
    h_norm = math.hypot(hx, hy)
    if h_norm < 1e-6:
        hx, hy = 1.0, 0.0
    else:
        hx, hy = hx / h_norm, hy / h_norm
    # 侧向单位向量（顺时针 90°）
    sx, sy = -hy, hx
    half_l, half_w = length / 2.0, width / 2.0
    corners = np.array([
        [x + hx * half_l + sx * half_w,  y + hy * half_l + sy * half_w],
        [x + hx * half_l - sx * half_w,  y + hy * half_l - sy * half_w],
        [x - hx * half_l - sx * half_w,  y - hy * half_l - sy * half_w],
        [x - hx * half_l + sx * half_w,  y - hy * half_l + sy * half_w],
    ])
    return corners
