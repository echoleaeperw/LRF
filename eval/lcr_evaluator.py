#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-Tail Coverage Rate (LCR) 评估器
=====================================

基于 Table 2 三指标同时满足（AND 逻辑）计算 LCR：

    LCR = |{ s ∈ S | TTC(s) ≤ τ_ttc  AND
                      TLC(s) ≤ τ_tlc  AND
                      THW(s) ≤ τ_thw }|  /  |S|

三个指标定义（Table 2，Long-tail Events 行）：
    I1: TTC (Time-to-Collision)      ≤ 1.5 s  (esp. ≤ 1.0 s)
    I2: TLC (Time-to-Lane-Crossing)  ≤ 0.8 s  (esp. ≤ 0.5–0.6 s)
    I3: THW (Time Headway)           ≤ 1.0 s  (esp. ≤ 0.5–0.7 s)

所有指标从保存的场景 JSON `fut_adv` 轨迹数据中重新计算，
不依赖生成阶段的文件夹分类（彻底解耦生成与评估）。

使用方式
--------
# 仅需场景目录：
result = compute_lcr_from_dir("out/run_xxx/scenario_results")

# 提供 NuScenesMapEnv 以获得准确 TLC（推荐）：
result = compute_lcr_from_dir("out/run_xxx/scenario_results",
                               map_env=map_env)

# CLI：
python eval/lcr_evaluator.py \\
    --scenario_dir out/run_xxx/scenario_results \\
    [--data_dir data/nuscenes/mini] \\
    [--ttc_tau 1.5] [--tlc_tau 0.8] [--thw_tau 1.0] \\
    [--strict]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 阈值配置
# ---------------------------------------------------------------------------

@dataclass
class LCRThresholds:
    """Table 2 长尾事件判定阈值（默认值 = 宽松档）"""
    ttc_s: float = 1.5   # TTC  ≤ this → 触发长尾
    tlc_s: float = 0.8   # TLC  ≤ this → 触发长尾
    thw_s: float = 1.0   # THW  ≤ this → 触发长尾

    # "esp." 严格档
    ttc_strict: float = 1.0
    tlc_strict: float = 0.6
    thw_strict: float = 0.7


# ---------------------------------------------------------------------------
# 场景指标结果
# ---------------------------------------------------------------------------

@dataclass
class SceneMetrics:
    scene_id:    str
    min_ttc:     float   # 秒，inf = 无碰撞风险
    min_tlc:     float   # 秒，inf = 无越道风险
    min_thw:     float   # 秒，inf = 无跟车风险
    is_longtail: bool    # 三项同时满足 AND 逻辑


# ---------------------------------------------------------------------------
# 底层向量工具
# ---------------------------------------------------------------------------

_EPS = 1e-6


def _velocities(traj: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """从位置差分计算速度向量和速度大小。

    参数
    ----
    traj : [T, 4]  (x, y, hx, hy)
    dt   : 时间步长 (s)

    返回
    ----
    vel   : [T-1, 2]  速度向量 (vx, vy)
    speed : [T-1]     速度大小
    """
    pos = traj[:, :2]
    vel = (pos[1:] - pos[:-1]) / dt
    speed = np.linalg.norm(vel, axis=1)
    return vel, speed


def _nearest_on_polyline(point: np.ndarray, poly: np.ndarray) -> Tuple[np.ndarray, float]:
    """返回折线上最近点及其距离。

    参数
    ----
    point : [2]
    poly  : [N, 2]

    返回
    ----
    nearest_pt : [2]
    min_dist   : float
    """
    min_dist = float("inf")
    nearest  = poly[0].copy()

    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab   = b - a
        len_sq = float(np.dot(ab, ab))
        if len_sq < 1e-10:
            proj = a
        else:
            t    = float(np.clip(np.dot(point - a, ab) / len_sq, 0.0, 1.0))
            proj = a + t * ab
        d = float(np.linalg.norm(point - proj))
        if d < min_dist:
            min_dist = d
            nearest  = proj

    return nearest, min_dist


# ---------------------------------------------------------------------------
# I1: TTC
# ---------------------------------------------------------------------------

def compute_min_ttc(ego_traj: np.ndarray,
                    atk_traj: np.ndarray,
                    dt:       float) -> float:
    """计算 ego 与攻击者之间的最小 TTC。

    考虑两种接近情形：
      Case A: attacker 在 ego 前方，ego 逼近（追尾）
      Case B: attacker 在 ego 后方，attacker 逼近（迎面/追及）

    参数
    ----
    ego_traj : [T, 4]  ego 对抗轨迹
    atk_traj : [T, 4]  攻击者对抗轨迹
    dt       : 时间步长 (s)

    返回
    ----
    min_ttc (s)，无有效值则返回 inf
    """
    T = min(len(ego_traj), len(atk_traj))
    if T < 2:
        return float("inf")

    ego = ego_traj[:T]
    atk = atk_traj[:T]

    ego_vel, _ = _velocities(ego, dt)   # [T-1, 2]
    atk_vel, _ = _velocities(atk, dt)   # [T-1, 2]

    # ego 朝向单位向量（取前 T-1 帧）
    ego_h = ego[:-1, 2:4]
    h_norm = np.linalg.norm(ego_h, axis=1, keepdims=True) + _EPS
    ego_h_unit = ego_h / h_norm          # [T-1, 2]

    # 纵向相对位置（attacker - ego，投影到 ego 朝向）
    rel_pos  = atk[:-1, :2] - ego[:-1, :2]           # [T-1, 2]
    d_lon    = np.einsum("ti,ti->t", rel_pos, ego_h_unit)  # [T-1]

    # 纵向相对速度（attacker - ego 在 ego 朝向上的投影）
    rel_vel  = atk_vel - ego_vel                          # [T-1, 2]
    v_rel    = np.einsum("ti,ti->t", rel_vel, ego_h_unit) # [T-1]

    ttc_list: List[float] = []

    # Case A: attacker 在前方(d_lon>0)，ego 正在接近(v_rel<0)
    mask_a = (d_lon > 0.0) & (v_rel < 0.0)
    if mask_a.any():
        ttc_list.extend((d_lon[mask_a] / (-v_rel[mask_a] + _EPS)).tolist())

    # Case B: attacker 在后方(d_lon<0)，正在追上(v_rel>0)
    mask_b = (d_lon < 0.0) & (v_rel > 0.0)
    if mask_b.any():
        ttc_list.extend((-d_lon[mask_b] / (v_rel[mask_b] + _EPS)).tolist())

    return float(np.min(ttc_list)) if ttc_list else float("inf")


# ---------------------------------------------------------------------------
# I3: THW
# ---------------------------------------------------------------------------

def compute_min_thw(ego_traj: np.ndarray,
                    atk_traj: np.ndarray,
                    dt:       float,
                    min_speed_ms: float = 0.5) -> float:
    """计算最小 THW (Time Headway)。

    THW = d_lon / v_ego
    仅在 attacker 位于 ego 正前方(d_lon > 0)且 ego 有行驶速度时有效。

    参数
    ----
    ego_traj     : [T, 4]
    atk_traj     : [T, 4]
    dt           : 时间步长 (s)
    min_speed_ms : ego 最低速度阈值（低于此视为静止）

    返回
    ----
    min_thw (s)，无有效值则返回 inf
    """
    T = min(len(ego_traj), len(atk_traj))
    if T < 2:
        return float("inf")

    ego = ego_traj[:T]
    atk = atk_traj[:T]

    _, ego_speed = _velocities(ego, dt)  # [T-1]

    ego_h = ego[:-1, 2:4]
    h_norm = np.linalg.norm(ego_h, axis=1, keepdims=True) + _EPS
    ego_h_unit = ego_h / h_norm

    rel_pos = atk[:-1, :2] - ego[:-1, :2]
    d_lon   = np.einsum("ti,ti->t", rel_pos, ego_h_unit)

    valid = (d_lon > 0.0) & (ego_speed > min_speed_ms)
    if not valid.any():
        return float("inf")

    thw_vals = d_lon[valid] / (ego_speed[valid] + _EPS)
    return float(np.min(thw_vals))


# ---------------------------------------------------------------------------
# I2: TLC
# ---------------------------------------------------------------------------

def compute_min_tlc(ego_traj: np.ndarray,
                    dt: float,
                    lane_boundaries: Optional[List[np.ndarray]] = None,
                    search_radius_m:   float = 50.0,
                    assumed_half_lane: float = 1.75,
                    min_vlat_ms:       float = 0.1) -> float:
    """计算最小 TLC (Time-to-Lane-Crossing)。

    TLC = d_lat / |v_lat|

    其中：
        d_lat = ego 到其正在靠近的车道边界的横向距离 (m)
        v_lat = ego 横向速度分量（垂直于朝向）

    有地图时（lane_boundaries 非 None）：
        从 NuScenesMapEnv._vector_cache[map]['lane_boundary'] 中
        取全局坐标折线，只保留 search_radius_m 范围内的边界，
        分别计算 ego 左右两侧距离，与运动方向匹配后求 TLC。

    无地图时：
        假设 ego 位于车道中央，单侧距离 = assumed_half_lane (m)。

    参数
    ----
    ego_traj        : [T, 4]
    dt              : 时间步长 (s)
    lane_boundaries : 全局坐标折线列表（每条为 np.ndarray [N,2]）
    search_radius_m : 搜索半径（米），过滤远处无关边界
    assumed_half_lane : 无地图时的单侧车道宽假设（米）
    min_vlat_ms     : 最低横向速度阈值（m/s），低于此忽略

    返回
    ----
    min_tlc (s)，无横向运动则返回 inf
    """
    T = len(ego_traj)
    if T < 2:
        return float("inf")

    ego_vel, _ = _velocities(ego_traj, dt)  # [T-1, 2]
    tlc_list: List[float] = []

    for t in range(T - 1):
        pos = ego_traj[t, :2]
        hx, hy = ego_traj[t, 2], ego_traj[t, 3]
        h_norm = np.sqrt(hx * hx + hy * hy) + _EPS
        hx, hy = hx / h_norm, hy / h_norm

        # 横向方向单位向量（ego 左侧为正）
        lat_x, lat_y = -hy, hx

        # 横向速度（有符号：正 = 向左，负 = 向右）
        vx, vy = ego_vel[t]
        v_lat = vx * lat_x + vy * lat_y

        if abs(v_lat) < min_vlat_ms:
            continue   # 横向速度不足，TLC 趋向无穷

        # ── 计算 ego 到车道边界的横向距离 ──────────────────────────────────
        if lane_boundaries is not None and len(lane_boundaries) > 0:
            # 筛选附近边界（避免遍历全图）
            nearby = [
                lb for lb in lane_boundaries
                if len(lb) >= 2
                and np.any(np.linalg.norm(lb - pos, axis=1) < search_radius_m)
            ]

            if nearby:
                # 对每条边界：计算最近点，再求"有符号横向距离"
                left_dists:  List[float] = []   # 边界在 ego 左侧
                right_dists: List[float] = []   # 边界在 ego 右侧

                for lb in nearby:
                    nearest_pt, abs_dist = _nearest_on_polyline(pos, lb)
                    # 最近点相对 ego 的横向符号
                    rel = nearest_pt - pos
                    sign = rel[0] * lat_x + rel[1] * lat_y
                    if sign >= 0:
                        left_dists.append(abs_dist)
                    else:
                        right_dists.append(abs_dist)

                # 与运动方向匹配
                if v_lat > 0 and left_dists:       # 向左运动 → 用左侧最近边界
                    d_to_boundary = min(left_dists)
                elif v_lat < 0 and right_dists:    # 向右运动 → 用右侧最近边界
                    d_to_boundary = min(right_dists)
                else:
                    # 运动方向侧无边界记录，退回全局最小距离
                    all_dists = left_dists + right_dists
                    d_to_boundary = min(all_dists) if all_dists else assumed_half_lane
            else:
                d_to_boundary = assumed_half_lane  # 附近无边界，用默认值
        else:
            # 无地图：假设 ego 在车道中央
            d_to_boundary = assumed_half_lane

        tlc = d_to_boundary / (abs(v_lat) + _EPS)
        tlc_list.append(tlc)

    return float(np.min(tlc_list)) if tlc_list else float("inf")


# ---------------------------------------------------------------------------
# 单场景评估
# ---------------------------------------------------------------------------

def evaluate_scene(scene_dict: Dict,
                   thresholds: LCRThresholds,
                   lane_boundaries: Optional[List[np.ndarray]] = None) -> SceneMetrics:
    """从场景字典计算三指标并判定是否为长尾。

    三指标对应的计算主体：
        I1 TTC  : ego  vs attacker（追尾 / 迎面）
        I2 TLC  : attacker 本身的越道时间（攻击者横向切入是长尾的核心行为）
        I3 THW  : ego  vs attacker（attacker 在 ego 前方时的时距）

    参数
    ----
    scene_dict      : 从 JSON 加载的场景字典
    thresholds      : LCRThresholds
    lane_boundaries : 该场景对应地图的车道边界折线列表（全局坐标）

    返回
    ----
    SceneMetrics
    """
    scene_id   = scene_dict.get("scene_id", "unknown")
    dt         = float(scene_dict.get("dt", 0.5))
    attack_agt = int(scene_dict.get("attack_agt", 1))

    fut_adv = np.array(scene_dict["fut_adv"], dtype=np.float32)  # [N, T, 4]
    N, T, _ = fut_adv.shape

    # 处理 NaN：插值填充（轨迹中偶尔出现的 NaN）
    for n in range(N):
        for dim in range(4):
            col = fut_adv[n, :, dim]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                fut_adv[n, :, dim] = 0.0
            elif nan_mask.any():
                idx = np.arange(T)
                fut_adv[n, :, dim] = np.interp(idx, idx[~nan_mask], col[~nan_mask])

    ego_traj = fut_adv[0]
    atk_traj = fut_adv[attack_agt] if attack_agt < N else fut_adv[min(1, N - 1)]

    # ── I1: TTC  (ego ↔ attacker) ──────────────────────────────────────────
    min_ttc = compute_min_ttc(ego_traj, atk_traj, dt)

    # ── I3: THW  (attacker 在 ego 前方时的时距) ────────────────────────────
    min_thw = compute_min_thw(ego_traj, atk_traj, dt)

    # ── I2: TLC  (攻击者越道时间) ─────────────────────────────────────────
    # 攻击者横向切入车道是长尾危险事件的核心，TLC 算在攻击者身上
    min_tlc = compute_min_tlc(atk_traj, dt, lane_boundaries=lane_boundaries)

    # ── AND 判定 ───────────────────────────────────────────────────────────
    is_longtail = (
        min_ttc <= thresholds.ttc_s and
        min_tlc <= thresholds.tlc_s and
        min_thw <= thresholds.thw_s
    )

    return SceneMetrics(
        scene_id    = scene_id,
        min_ttc     = min_ttc,
        min_tlc     = min_tlc,
        min_thw     = min_thw,
        is_longtail = is_longtail,
    )


# ---------------------------------------------------------------------------
# 批量评估 + LCR 汇总
# ---------------------------------------------------------------------------

def compute_lcr_from_dir(
        scenario_dir: str,
        thresholds:   Optional[LCRThresholds] = None,
        map_env:      Optional[object]        = None,
        verbose:      bool                    = True,
        max_scenes:   Optional[int]           = None,
) -> Dict:
    """扫描 scenario_results/ 目录，重新计算三指标并输出 LCR。

    目录结构预期：
        scenario_dir/
            longtail_condition/   ← 生成时的旧分类（本函数不依赖此分类）
            high_risk/
            low_risk/

    参数
    ----
    scenario_dir : scenario_results/ 目录路径
    thresholds   : LCRThresholds（默认使用 Table 2 宽松阈值）
    map_env      : NuScenesMapEnv 实例（可选，用于准确 TLC）
    verbose      : 是否打印每条场景结果

    返回
    ----
    {
        "lcr":          float,   # 主指标
        "lcr_strict":   float,   # 严格阈值 LCR
        "total":        int,
        "longtail_count": int,
        "mean_ttc":     float,
        "mean_tlc":     float,
        "mean_thw":     float,
        "per_scene":    List[dict],   # 每条场景详情
        "breakdown":    dict,         # 各单指标达标率
    }
    """
    if thresholds is None:
        thresholds = LCRThresholds()

    thresholds_strict = LCRThresholds(
        ttc_s=thresholds.ttc_strict,
        tlc_s=thresholds.tlc_strict,
        thw_s=thresholds.thw_strict,
    )

    # 收集所有 JSON 场景文件（不区分子目录名称）
    json_files: List[str] = []
    for risk_dir in ["longtail_condition", "high_risk", "low_risk"]:
        pattern = os.path.join(scenario_dir, risk_dir, "*.json")
        json_files.extend(sorted(glob.glob(pattern)))
    # 也兼容直接放在 scenario_dir 下的情况
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(scenario_dir, "**", "*.json"),
                                      recursive=True))

    if not json_files:
        print(f"[LCR] 未找到任何 JSON 场景文件于 {scenario_dir}")
        return {"lcr": 0.0, "total": 0}

    # 与 run_adversarial_evaluation.py 保持一致：取最后 max_scenes 个（最新生成的）
    if max_scenes is not None and len(json_files) > max_scenes:
        json_files = sorted(json_files)[-max_scenes:]
        print(f"[LCR] max_scenes={max_scenes}，从 {len(json_files)+len(json_files)-max_scenes} 个场景中取后 {max_scenes} 个")

    per_scene_results: List[SceneMetrics] = []
    per_scene_dicts:   List[Dict]         = []
    n_loaded = 0

    for fpath in json_files:
        try:
            with open(fpath) as f:
                scene = json.load(f)
        except Exception as e:
            print(f"[LCR] 读取失败 {fpath}: {e}")
            continue

        # 若提供了 map_env，获取对应地图的车道边界
        lane_boundaries: Optional[List[np.ndarray]] = None
        if map_env is not None:
            map_name = scene.get("map", "")
            try:
                vec_cache = map_env._vector_cache.get(map_name, {})
                raw_boundaries = vec_cache.get("lane_boundary", [])
                # 转为 np.ndarray 列表（有些已经是 ndarray，有些是 list）
                lane_boundaries = [
                    np.asarray(lb, dtype=np.float32) for lb in raw_boundaries
                    if len(lb) >= 2
                ]
            except Exception as e:
                print(f"[LCR] 获取地图边界失败 ({map_name}): {e}")
                lane_boundaries = None

        # 为 scene_id 字段赋值（用文件名）
        if "scene_id" not in scene:
            scene["scene_id"] = os.path.splitext(os.path.basename(fpath))[0]

        try:
            metrics = evaluate_scene(scene, thresholds, lane_boundaries)
            metrics_strict = evaluate_scene(scene, thresholds_strict, lane_boundaries)
        except Exception as e:
            print(f"[LCR] 评估失败 {fpath}: {e}")
            continue

        per_scene_results.append(metrics)
        per_scene_dicts.append({
            "scene_id":      metrics.scene_id,
            "file":          fpath,
            "min_ttc":       round(metrics.min_ttc, 4),
            "min_tlc":       round(metrics.min_tlc, 4),
            "min_thw":       round(metrics.min_thw, 4),
            "is_longtail":   metrics.is_longtail,
            "is_longtail_strict": metrics_strict.is_longtail,
            "ttc_ok":  metrics.min_ttc <= thresholds.ttc_s,
            "tlc_ok":  metrics.min_tlc <= thresholds.tlc_s,
            "thw_ok":  metrics.min_thw <= thresholds.thw_s,
        })
        n_loaded += 1

        if verbose:
            flag = "✓ LONGTAIL" if metrics.is_longtail else "  normal  "
            ttc_str = f"{metrics.min_ttc:.2f}" if metrics.min_ttc < 999 else "  ∞ "
            tlc_str = f"{metrics.min_tlc:.2f}" if metrics.min_tlc < 999 else "  ∞ "
            thw_str = f"{metrics.min_thw:.2f}" if metrics.min_thw < 999 else "  ∞ "
            print(f"  [{flag}] {metrics.scene_id:15s}  "
                  f"TTC={ttc_str}s  TLC={tlc_str}s  THW={thw_str}s")

    total    = len(per_scene_results)
    if total == 0:
        return {"lcr": 0.0, "total": 0}

    longtail_count        = sum(1 for m in per_scene_results if m.is_longtail)
    longtail_count_strict = sum(1 for d in per_scene_dicts   if d["is_longtail_strict"])
    lcr        = longtail_count        / total
    lcr_strict = longtail_count_strict / total

    ttc_ok_rate = sum(1 for d in per_scene_dicts if d["ttc_ok"]) / total
    tlc_ok_rate = sum(1 for d in per_scene_dicts if d["tlc_ok"]) / total
    thw_ok_rate = sum(1 for d in per_scene_dicts if d["thw_ok"]) / total

    finite_ttc = [m.min_ttc for m in per_scene_results if m.min_ttc < 999]
    finite_tlc = [m.min_tlc for m in per_scene_results if m.min_tlc < 999]
    finite_thw = [m.min_thw for m in per_scene_results if m.min_thw < 999]

    result = {
        # ── 主指标 ──────────────────────────────────────────────────────────
        "lcr":              round(lcr, 4),
        "lcr_strict":       round(lcr_strict, 4),
        "total":            total,
        "longtail_count":   longtail_count,
        "longtail_count_strict": longtail_count_strict,

        # ── 各指标均值 ───────────────────────────────────────────────────────
        "mean_ttc": round(float(np.mean(finite_ttc)), 4) if finite_ttc else float("inf"),
        "mean_tlc": round(float(np.mean(finite_tlc)), 4) if finite_tlc else float("inf"),
        "mean_thw": round(float(np.mean(finite_thw)), 4) if finite_thw else float("inf"),

        # ── 单指标达标率（用于诊断哪个指标是瓶颈）──────────────────────────
        "breakdown": {
            "ttc_rate": round(ttc_ok_rate, 4),
            "tlc_rate": round(tlc_ok_rate, 4),
            "thw_rate": round(thw_ok_rate, 4),
        },

        # ── 使用的阈值 ───────────────────────────────────────────────────────
        "thresholds": {
            "ttc_s": thresholds.ttc_s,
            "tlc_s": thresholds.tlc_s,
            "thw_s": thresholds.thw_s,
        },

        # ── 每场景详情 ───────────────────────────────────────────────────────
        "per_scene": per_scene_dicts,
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(result: Dict, thresholds: LCRThresholds) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Long-Tail Coverage Rate (LCR) — Table 2 AND 逻辑")
    print(sep)
    print(f"  阈值：TTC ≤ {thresholds.ttc_s}s  |  TLC ≤ {thresholds.tlc_s}s  |  THW ≤ {thresholds.thw_s}s")
    print(f"  总场景数  : {result['total']}")
    print(f"  长尾场景数: {result['longtail_count']}")
    print(f"  LCR       : {result['lcr']:.4f}  ({result['lcr'] * 100:.1f}%)")
    print(f"  LCR (严格): {result['lcr_strict']:.4f}  ({result['lcr_strict'] * 100:.1f}%)")
    print()
    print("  各单指标达标率（AND 瓶颈分析）：")
    bd = result["breakdown"]
    print(f"    TTC ≤ {thresholds.ttc_s}s  : {bd['ttc_rate'] * 100:.1f}%")
    print(f"    TLC ≤ {thresholds.tlc_s}s  : {bd['tlc_rate'] * 100:.1f}%  "
          + ("⚠ 无地图，使用代理值" if result.get("tlc_proxy") else ""))
    print(f"    THW ≤ {thresholds.thw_s}s  : {bd['thw_rate'] * 100:.1f}%")
    print()
    print("  指标均值（有效帧）：")
    print(f"    mean TTC = {result['mean_ttc']} s")
    print(f"    mean TLC = {result['mean_tlc']} s")
    print(f"    mean THW = {result['mean_thw']} s")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算 LCR（Long-Tail Coverage Rate）— Table 2 AND 逻辑"
    )
    parser.add_argument(
        "--scenario_dir", required=True,
        help="scenario_results/ 目录路径（含 longtail_condition/high_risk/low_risk 子目录）"
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="NuScenes 数据目录（提供则用真实地图计算 TLC，否则使用代理值）"
    )
    parser.add_argument("--data_version", default="mini",
                        choices=["mini", "trainval"])
    parser.add_argument("--ttc_tau",  type=float, default=1.5,
                        help="TTC  长尾阈值 (s)，默认 1.5")
    parser.add_argument("--tlc_tau",  type=float, default=0.8,
                        help="TLC  长尾阈值 (s)，默认 0.8")
    parser.add_argument("--thw_tau",  type=float, default=1.0,
                        help="THW  长尾阈值 (s)，默认 1.0")
    parser.add_argument("--strict",   action="store_true",
                        help="同时报告严格档（esp.）LCR")
    parser.add_argument("--quiet",    action="store_true",
                        help="不打印逐场景结果")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="最多评估的场景数（取最后 N 个，与 run_adversarial_evaluation 一致）")
    parser.add_argument("--out_json", default=None,
                        help="将结果输出到 JSON 文件")
    args = parser.parse_args()

    thresholds = LCRThresholds(
        ttc_s = args.ttc_tau,
        tlc_s = args.tlc_tau,
        thw_s = args.thw_tau,
    )

    # 可选：加载地图以获得精确 TLC（使用轻量的 NuScenesMap，不依赖 NuScenesMapEnv）
    map_env = None
    if args.data_dir is not None:
        try:
            # Script lives in eval/; add repo root + src/ to sys.path for downstream imports.
            _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            for _p in (_repo_root, os.path.join(_repo_root, "src")):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            from nuscenes.map_expansion.map_api import NuScenesMap
            from nuscenes.map_expansion.arcline_path_utils import discretize_lane

            # NuScenesMap(dataroot=...) 需要包含 maps/ 的顶层目录
            for data_path in [args.data_dir,
                               os.path.join(args.data_dir, args.data_version)]:
                if os.path.isdir(os.path.join(data_path, "maps")):
                    break
            print(f"[LCR] 加载地图数据：{data_path}")

            NUSC_MAP_SIZES = {
                'singapore-onenorth':        [2025.0, 1585.6],
                'singapore-hollandvillage':  [2922.9, 2808.3],
                'singapore-queenstown':      [3687.1, 3228.6],
                'boston-seaport':            [2118.1, 2979.5],
            }
            HALF_LANE_W = 1.8
            FLIP_MAPS   = {'singapore-onenorth', 'singapore-hollandvillage',
                            'singapore-queenstown'}

            def _offset_polyline(pts, d):
                """将折线向左（d>0）或右（d<0）偏移 |d| 米。"""
                if len(pts) < 2:
                    return pts.copy()
                tangents = np.diff(pts, axis=0)
                tangents = np.vstack([tangents, tangents[-1]])
                norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
                tangents = tangents / norms
                normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
                return pts + d * normals

            vector_cache: Dict[str, List[np.ndarray]] = {}
            map_names = list(NUSC_MAP_SIZES.keys())
            for mname in map_names:
                try:
                    nmap = NuScenesMap(dataroot=data_path, map_name=mname)
                except Exception:
                    continue
                mheight = NUSC_MAP_SIZES[mname][0]
                is_sg   = mname in FLIP_MAPS
                boundaries: List[np.ndarray] = []
                for lt in [rec['token'] for rec in nmap.lane]:
                    try:
                        arcline = nmap.get_arcline_path(lt)
                        pts3d   = np.array(discretize_lane(arcline, resolution_meters=1.0))
                        pts2d   = pts3d[:, :2].copy()
                        if is_sg:
                            pts2d[:, 1] = mheight - pts2d[:, 1]
                        if len(pts2d) >= 2:
                            boundaries.append(_offset_polyline(pts2d,  HALF_LANE_W))
                            boundaries.append(_offset_polyline(pts2d, -HALF_LANE_W))
                    except Exception:
                        pass
                vector_cache[mname] = boundaries
                print(f"  {mname}: {len(boundaries)} 条车道边界")

            # 封装成与 compute_lcr_from_dir 期望格式相同的对象
            class _LightMapEnv:
                def __init__(self, cache):
                    self._vector_cache = {k: {"lane_boundary": v}
                                          for k, v in cache.items()}
            map_env = _LightMapEnv(vector_cache)
            print("[LCR] 地图加载完成，TLC 将使用真实车道边界")
        except Exception as e:
            print(f"[LCR] 地图加载失败（{e}），TLC 改用代理值")
            map_env = None

    print(f"\n[LCR] 评估目录：{args.scenario_dir}")
    print(f"[LCR] 阈值：TTC≤{thresholds.ttc_s}s  TLC≤{thresholds.tlc_s}s  THW≤{thresholds.thw_s}s\n")

    result = compute_lcr_from_dir(
        args.scenario_dir,
        thresholds=thresholds,
        map_env=map_env,
        verbose=not args.quiet,
        max_scenes=args.max_scenes,
    )
    result["tlc_proxy"] = (map_env is None)

    _print_summary(result, thresholds)

    if args.out_json:
        # 将 per_scene 截断以控制输出大小
        out = {k: v for k, v in result.items() if k != "per_scene"}
        out["per_scene_count"] = len(result.get("per_scene", []))
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[LCR] 结果已写入 {args.out_json}")


if __name__ == "__main__":
    main()
