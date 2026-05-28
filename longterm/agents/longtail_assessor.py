"""
LongTailPotentialAssessor — 长尾场景潜力评估器

在 AnalysisAgent 行为识别后、权重生成前插入（flow.py Step 1.5）。
用确定性规则检验当前场景是否具备生成特定长尾事件的前提条件，
输出可控的权重调节系数（乘性修正，叠加在 ReflectionAgent 基础权重之上）。

三种目标长尾类型
─────────────────
  1. SuddenBrakingUnpredictable  — 前车无征兆急刹
  2. HesitantCutIn               — 邻车犹豫后突然并线
  3. OccludedConflict            — 对向来车遮挡下的异常让行/抢行

评估逻辑
─────────────────
  potential_score = (满足前提条件数/总条件数) × 满足条件的平均置信度
  score ≥ min_potential (默认0.5) 时输出有效修正，否则返回空修正。
"""

import json
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from rich import print

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Long-tail subtype definitions
# ──────────────────────────────────────────────────────────────────────────────

_LONGTAIL_SUBTYPES: Dict[str, Dict] = {
    "SuddenBrakingUnpredictable": {
        "description": "前车无征兆急刹：前车在无明显减速迹象时突然制动，压缩Ego反应时间至极限",
        "parent_behavior": "SuddenBraking",
        "preconditions": [
            "has_leading_vehicle",       # 前方同车道存在车辆
            "thw_in_range",              # 跟车时距 0.8~3.5s（危险但有反应空间）
            "ego_speed_sufficient",      # ego速度 ≥ 7.0 m/s (~25 km/h)
            "leader_no_decel_trend",     # 前车近期无减速趋势（< 1.5 m/s²）
        ],
        # 乘性修正系数，叠加于 ReflectionAgent 基础权重
        # motion_prior_atk 保守修正：只轻微放松以允许急刹偏离先验，
        # 保持在 0.0005 以上确保轨迹物理合理（reflection.py 设有硬下限 0.0004）
        "weight_modifiers": {
            "adv_crash":        1.6,   # 强化朝向碰撞的驱动
            "ttc":              2.0,   # 纵向TTC是主要指标
            "min_dist_lat":     0.5,   # 纵向碰撞，横向不关键
            "yaw_rate":         0.4,   # 急刹场景横摆率仍保持低位
            "motion_prior_atk": 0.55,  # 适度放松：允许急刹偏离先验，但保持轨迹真实性
            "init_z_atk":       0.6,
        },
    },

    "HesitantCutIn": {
        "description": "邻车犹豫后突然并线：相邻车道车辆横向摆动后突然切入，造成认知欺骗性",
        "parent_behavior": "AggressiveCutIn",
        "preconditions": [
            "has_adjacent_vehicle",          # 相邻车道存在车辆
            "adjacent_lateral_oscillation",  # 该车横向位置有摆动迹象（std > 0.4m）
            "adjacent_in_merging_zone",      # 纵向处于可并线区域（< 15m）
            "lateral_gap_available",         # ego侧有横向空间（无紧贴障碍）
        ],
        # min_dist_lat 改为 1.5（原 2.5）：避免横向约束完全主导优化导致碰撞率下降
        "weight_modifiers": {
            "adv_crash":        1.5,
            "min_dist_lat":     1.5,   # 适度强化横向，不过度主导
            "yaw_rate":         1.8,   # 强化横摆（在新量级下依然有意义）
            "ttc":              1.3,
            "motion_prior_atk": 0.5,   # 允许非常规横向轨迹，但保持可信的摆动→切入轨迹形态
            "init_z_atk":       0.55,
        },
    },

    "OccludedConflict": {
        "description": "对向来车遮挡下的异常让行/抢行：路口场景下第三方遮挡导致意外冲突",
        "parent_behavior": "IntersectionRush",
        "preconditions": [
            "near_intersection",              # 接近或位于路口
            "has_opposing_vehicle",           # 存在对向来车（可能形成遮挡）
            "has_third_conflicting_vehicle",  # 存在第三辆车构成让行/抢行冲突
            "complex_scene",                  # 场景复杂度足够（≥3辆车）
        ],
        "weight_modifiers": {
            "adv_crash":        1.8,
            "ttc":              2.0,
            "min_dist_lat":     1.8,
            "yaw_rate":         1.6,   # 路口需要转向，适度强化
            "motion_prior_atk": 0.6,   # 路口场景允许偏离常规，但仍保持自然的路口行驶形态
            "init_z_atk":       0.65,
            "coll_veh":         0.75,  # 适度放松多车约束，允许路口复杂博弈
        },
    },
}

# 父行为 → 匹配的长尾子类型（供按行为标签排序候选用）
_PARENT_BEHAVIOR_MAP: Dict[str, List[str]] = {
    "SuddenBrakingUnpredictable": ["SuddenBraking", "AggressiveTailgating"],
    "HesitantCutIn":              ["AggressiveCutIn", "LaneDeparture"],
    "OccludedConflict":           ["IntersectionRush", "MultiVehiclePincer"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _lon_lat_dist(
    ego_x: float, ego_y: float,
    ref_x: float, ref_y: float,
    ego_heading_deg: float,
) -> Tuple[float, float]:
    """
    Compute longitudinal and lateral distance from ego to reference vehicle,
    projected along ego's heading direction.

    Returns:
        lon: positive = in front, negative = behind
        lat: always positive (absolute value)
    """
    dx = ref_x - ego_x
    dy = ref_y - ego_y
    h_rad = math.radians(ego_heading_deg)
    fwd_x, fwd_y = math.cos(h_rad), math.sin(h_rad)
    lon = dx * fwd_x + dy * fwd_y
    lat = abs(-dx * fwd_y + dy * fwd_x)
    return lon, lat


def _heading_diff_deg(h1_deg: float, h2_deg: float) -> float:
    """Absolute angular difference between two headings in [0, 180] degrees."""
    diff = abs((h1_deg - h2_deg + 540) % 360 - 180)
    return diff


def _get_current_state(trajectory: List[Dict]) -> Optional[Dict]:
    """Return the trajectory point closest to t=0 (current moment)."""
    if not trajectory:
        return None
    return min(trajectory, key=lambda p: abs(p.get("t", 99.0)))


def _get_past_points(trajectory: List[Dict], window: float = 3.0) -> List[Dict]:
    """Return past trajectory points (t ≤ 0) within the given time window."""
    pts = [p for p in trajectory if -window <= p.get("t", 0.0) <= 0]
    return sorted(pts, key=lambda p: p.get("t", 0.0))


# ──────────────────────────────────────────────────────────────────────────────
# Precondition checker — all checks return (met, confidence, detail_str)
# ──────────────────────────────────────────────────────────────────────────────

class _PreconditionChecker:
    """
    Stateful checker initialised with parsed scenario data.
    Each check_* method returns (met: bool, confidence: float, detail: str).
    confidence ∈ [0, 1] reflects how strongly the condition is satisfied.
    """

    def __init__(self, scenario: Dict) -> None:
        self._scenario = scenario
        self._vehicles: List[Dict] = scenario.get("vehicles", [])
        self._ego: Optional[Dict] = None
        self._ego_state: Optional[Dict] = None
        self._non_ego: List[Dict] = []

        for v in self._vehicles:
            if v.get("is_ego", False) or v.get("id", -1) == 0:
                self._ego = v
            else:
                self._non_ego.append(v)

        if self._ego:
            self._ego_state = _get_current_state(self._ego.get("trajectory", []))

    # ── SuddenBrakingUnpredictable preconditions ───────────────────────────

    def check_has_leading_vehicle(self) -> Tuple[bool, float, str]:
        """前车存在：正前方同车道（纵向 2~60m，横向 < 3m）"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        ex, ey = self._ego_state["x"], self._ego_state["y"]
        eh = self._ego_state.get("heading", 0.0)
        best = None
        for v in self._non_ego:
            vs = _get_current_state(v.get("trajectory", []))
            if not vs:
                continue
            lon, lat = _lon_lat_dist(ex, ey, vs["x"], vs["y"], eh)
            if 2.0 < lon < 60.0 and lat < 3.0:
                if best is None or lon < best[0]:
                    best = (lon, lat, v.get("id", "?"))
        if best:
            lon, lat, vid = best
            conf = 1.0 - lon / 60.0
            return True, round(conf, 3), f"vehicle_{vid} lon={lon:.1f}m lat={lat:.1f}m"
        return False, 0.0, "no leading vehicle found"

    def check_thw_in_range(
        self, thw_min: float = 0.8, thw_max: float = 3.5
    ) -> Tuple[bool, float, str]:
        """跟车时距（THW = lon / ego_vel）在 0.8~3.5s 之间"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        ego_vel = self._ego_state.get("velocity", 0.0)
        if ego_vel < 1e-3:
            return False, 0.0, "ego is stationary"
        ex, ey = self._ego_state["x"], self._ego_state["y"]
        eh = self._ego_state.get("heading", 0.0)
        for v in self._non_ego:
            vs = _get_current_state(v.get("trajectory", []))
            if not vs:
                continue
            lon, lat = _lon_lat_dist(ex, ey, vs["x"], vs["y"], eh)
            if 2.0 < lon < 60.0 and lat < 3.0:
                thw = lon / ego_vel
                in_range = thw_min <= thw <= thw_max
                mid = (thw_min + thw_max) / 2.0
                half_span = (thw_max - thw_min) / 2.0 + 1e-3
                conf = max(0.0, 1.0 - abs(thw - mid) / half_span)
                return in_range, round(conf, 3), f"THW={thw:.2f}s"
        return False, 0.0, "no leading vehicle for THW calc"

    def check_ego_speed_sufficient(self, min_speed_ms: float = 7.0) -> Tuple[bool, float, str]:
        """Ego 当前速度 ≥ 7.0 m/s (~25 km/h)"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        vel = self._ego_state.get("velocity", 0.0)
        met = vel >= min_speed_ms
        conf = min(vel / (min_speed_ms * 2.0), 1.0)
        return met, round(conf, 3), f"ego_vel={vel:.1f}m/s"

    def check_leader_no_decel_trend(self, max_decel: float = 1.5) -> Tuple[bool, float, str]:
        """前车近期无减速趋势（平均减速度 < 1.5 m/s²）"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        ex, ey = self._ego_state["x"], self._ego_state["y"]
        eh = self._ego_state.get("heading", 0.0)
        for v in self._non_ego:
            vs = _get_current_state(v.get("trajectory", []))
            if not vs:
                continue
            lon, lat = _lon_lat_dist(ex, ey, vs["x"], vs["y"], eh)
            if 2.0 < lon < 60.0 and lat < 3.0:
                past_pts = _get_past_points(v.get("trajectory", []), window=3.0)
                if len(past_pts) < 2:
                    return True, 0.6, "insufficient history, assume stable speed"
                vels = [p.get("velocity", 0.0) for p in past_pts]
                times = [p.get("t", 0.0) for p in past_pts]
                dt = times[-1] - times[0]
                if abs(dt) < 1e-3:
                    return True, 0.5, "dt too small"
                decel_rate = -(vels[-1] - vels[0]) / dt  # positive = decelerating
                no_decel = decel_rate < max_decel
                conf = max(0.0, 1.0 - decel_rate / (max_decel * 2.0))
                return no_decel, round(conf, 3), f"decel_rate={decel_rate:.2f}m/s²"
        return False, 0.0, "leading vehicle not found for decel check"

    # ── HesitantCutIn preconditions ────────────────────────────────────────

    def check_has_adjacent_vehicle(
        self, lat_min: float = 1.5, lat_max: float = 6.0, lon_range: float = 20.0
    ) -> Tuple[bool, float, str]:
        """相邻车道存在车辆（横向 1.5~6m，纵向距离 ≤ 20m）"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        ex, ey = self._ego_state["x"], self._ego_state["y"]
        eh = self._ego_state.get("heading", 0.0)
        for v in self._non_ego:
            vs = _get_current_state(v.get("trajectory", []))
            if not vs:
                continue
            lon, lat = _lon_lat_dist(ex, ey, vs["x"], vs["y"], eh)
            if abs(lon) <= lon_range and lat_min <= lat <= lat_max:
                conf = 1.0 - (lat - lat_min) / (lat_max - lat_min + 1e-3)
                return True, round(conf, 3), f"vehicle_{v.get('id')} lon={lon:.1f}m lat={lat:.1f}m"
        return False, 0.0, "no adjacent vehicle"

    def check_adjacent_lateral_oscillation(
        self, std_threshold: float = 0.4
    ) -> Tuple[bool, float, str]:
        """相邻车辆横向摆动（标准差 > 0.4m，犹豫迹象）"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        ex, ey = self._ego_state["x"], self._ego_state["y"]
        eh = self._ego_state.get("heading", 0.0)
        for v in self._non_ego:
            vs = _get_current_state(v.get("trajectory", []))
            if not vs:
                continue
            lon, lat = _lon_lat_dist(ex, ey, vs["x"], vs["y"], eh)
            if abs(lon) <= 20.0 and 1.5 <= lat <= 6.0:
                past_pts = _get_past_points(v.get("trajectory", []), window=4.0)
                if len(past_pts) < 3:
                    return False, 0.3, "insufficient history for oscillation check"
                h_rad = math.radians(eh)
                lat_positions = [
                    -p["x"] * math.sin(h_rad) + p["y"] * math.cos(h_rad)
                    for p in past_pts
                ]
                mean_lat = sum(lat_positions) / len(lat_positions)
                variance = sum((l - mean_lat) ** 2 for l in lat_positions) / len(lat_positions)
                std = math.sqrt(variance)
                met = std > std_threshold
                conf = min(std / (std_threshold * 2.0), 1.0)
                return met, round(conf, 3), f"lateral_std={std:.3f}m"
        return False, 0.0, "no adjacent vehicle for oscillation check"

    def check_adjacent_in_merging_zone(self, lon_range: float = 15.0) -> Tuple[bool, float, str]:
        """相邻车辆在可并线区域（纵向距离 < 15m）"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        ex, ey = self._ego_state["x"], self._ego_state["y"]
        eh = self._ego_state.get("heading", 0.0)
        for v in self._non_ego:
            vs = _get_current_state(v.get("trajectory", []))
            if not vs:
                continue
            lon, lat = _lon_lat_dist(ex, ey, vs["x"], vs["y"], eh)
            if abs(lon) <= lon_range and 1.5 <= lat <= 6.0:
                conf = 1.0 - abs(lon) / lon_range
                return True, round(conf, 3), f"vehicle_{v.get('id')} lon={lon:.1f}m"
        return False, 0.0, "adjacent vehicle not in merging zone"

    def check_lateral_gap_available(self) -> Tuple[bool, float, str]:
        """Ego 侧有足够横向空间（相邻车横向 ≥ 2.5m，说明尚未紧贴）"""
        met, conf, detail = self.check_has_adjacent_vehicle(lat_min=2.5, lat_max=6.0)
        return met, conf, f"lateral_gap: {detail}"

    # ── OccludedConflict preconditions ─────────────────────────────────────

    def check_near_intersection(self) -> Tuple[bool, float, str]:
        """场景位于或接近路口"""
        for v in self._vehicles:
            ctx = v.get("map_context", {})
            if ctx.get("in_intersection", False):
                return True, 1.0, f"vehicle_{v.get('id')} in_intersection=True"
        rel_text = self._scenario.get("relative_motion_analysis", "")
        dyn_text = self._scenario.get("dynamic_analysis", {}).get("traffic_flow", "")
        for kw in ["intersection", "路口", "junction", "crossroad", "交叉"]:
            if kw.lower() in rel_text.lower() or kw.lower() in dyn_text.lower():
                return True, 0.7, f"keyword '{kw}' in scene description"
        return False, 0.0, "no intersection evidence"

    def check_has_opposing_vehicle(self, heading_diff_min: float = 120.0) -> Tuple[bool, float, str]:
        """存在对向来车（航向差 ≥ 120°）"""
        if not self._ego_state:
            return False, 0.0, "no ego state"
        ego_hdg = self._ego_state.get("heading", 0.0)
        for v in self._non_ego:
            vs = _get_current_state(v.get("trajectory", []))
            if not vs:
                continue
            h_diff = _heading_diff_deg(ego_hdg, vs.get("heading", 0.0))
            if h_diff >= heading_diff_min:
                conf = (h_diff - heading_diff_min) / (180.0 - heading_diff_min + 1e-3)
                return True, round(min(conf, 1.0), 3), f"vehicle_{v.get('id')} heading_diff={h_diff:.0f}°"
        return False, 0.0, "no opposing vehicle (heading diff < 120°)"

    def check_has_third_conflicting_vehicle(self) -> Tuple[bool, float, str]:
        """存在第三辆车（场景共 ≥ 3 辆，包含ego）"""
        n = len(self._vehicles)
        met = n >= 3
        conf = min((n - 2) / 3.0, 1.0) if n >= 3 else 0.0
        return met, round(conf, 3), f"total_vehicles={n}"

    def check_complex_scene(self, min_vehicles: int = 3) -> Tuple[bool, float, str]:
        """场景复杂度足够（车辆数 ≥ 3 或 dynamic_analysis 标注中/高复杂度）"""
        n = len(self._vehicles)
        if n >= min_vehicles:
            return True, round(min(n / 5.0, 1.0), 3), f"vehicle_count={n}"
        complexity = self._scenario.get("dynamic_analysis", {}).get("complexity", "")
        for kw in ["中等", "高", "medium", "high", "complex"]:
            if kw in complexity:
                return True, 0.7, f"complexity='{complexity}'"
        return False, 0.0, f"vehicle_count={n}, complexity='{complexity}'"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class LongTailPotentialAssessor:
    """
    长尾场景潜力评估器（确定性规则，无 LLM）

    assess() 接收解析后的场景 JSON 和 AnalysisAgent 的行为标签，
    返回最匹配的长尾子类型及可控的权重修正系数。

    potential_score 计算
    ─────────────────────
        score = (满足条件数 / 总条件数) × 满足条件的平均置信度

    score ≥ min_potential 时输出有效修正；否则返回空修正，
    不干扰 ReflectionAgent 的基础权重。
    """

    def __init__(self, min_potential: float = 0.5) -> None:
        self.min_potential = min_potential

    def assess(
        self,
        scenario_data: Any,
        behavior_label: str = "",
    ) -> Dict[str, Any]:
        """
        评估场景的长尾潜力。

        Args:
            scenario_data: 场景 JSON（str 或 dict，来自 ScenarioExtractor）
            behavior_label: AnalysisAgent 识别的行为标签（用于候选排序）

        Returns:
            {
                "longtail_subtype":     str,               # 最匹配的长尾子类型，空表示无
                "potential_score":      float,             # 0.0~1.0
                "met_preconditions":    List[str],
                "failed_preconditions": List[str],
                "weight_modifiers":     Dict[str, float],  # 乘性修正系数
                "assessment_summary":   str,               # 可读摘要
            }
        """
        if isinstance(scenario_data, str):
            try:
                scenario = json.loads(scenario_data)
            except (json.JSONDecodeError, ValueError) as exc:
                return self._empty_result(f"JSON parse error: {exc}")
        elif isinstance(scenario_data, dict):
            scenario = scenario_data
        else:
            return self._empty_result(f"unsupported type: {type(scenario_data)}")

        checker = _PreconditionChecker(scenario)
        candidates = self._rank_candidates(behavior_label)

        best_name: str = ""
        best_score: float = -1.0
        best_data: Optional[tuple] = None

        for subtype_name in candidates:
            subtype_def = _LONGTAIL_SUBTYPES[subtype_name]
            score, met, failed, details = self._evaluate_preconditions(
                checker, subtype_def["preconditions"]
            )
            if score > best_score:
                best_score = score
                best_name = subtype_name
                best_data = (met, failed, details, subtype_def)

        if best_data is None:
            return self._empty_result("no candidates evaluated")

        met, failed, details, subtype_def = best_data

        if best_score < self.min_potential:
            return self._empty_result(
                f"best_score={best_score:.2f} < min_potential={self.min_potential:.2f} "
                f"(candidate: {best_name})"
            )

        summary = (
            f"[LTA] 子类型: {best_name} | score={best_score:.2f} "
            f"(parent={subtype_def['parent_behavior']})\n"
            f"  描述: {subtype_def['description']}\n"
            f"  满足条件({len(met)}): {', '.join(met) if met else 'none'}\n"
            f"  未满足({len(failed)}): {', '.join(failed) if failed else 'none'}\n"
            f"  细节: {details}"
        )

        print(f"[bold green]{summary}[/bold green]")

        return {
            "longtail_subtype":     best_name,
            "potential_score":      round(best_score, 3),
            "met_preconditions":    met,
            "failed_preconditions": failed,
            "weight_modifiers":     subtype_def["weight_modifiers"].copy(),
            "assessment_summary":   summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_candidates(behavior_label: str) -> List[str]:
        """
        Return long-tail subtype names ordered by alignment with behavior_label.
        Compatible subtypes (same parent) are listed first.
        """
        compatible, others = [], []
        for subtype_name, parents in _PARENT_BEHAVIOR_MAP.items():
            if behavior_label in parents:
                compatible.append(subtype_name)
            else:
                others.append(subtype_name)
        return compatible + others

    @staticmethod
    def _evaluate_preconditions(
        checker: _PreconditionChecker,
        preconditions: List[str],
    ) -> Tuple[float, List[str], List[str], str]:
        """
        Run all precondition checks for a given subtype.

        Returns:
            score:   potential score ∈ [0, 1]
            met:     list of satisfied condition names
            failed:  list of unsatisfied condition names
            details: human-readable per-condition string
        """
        check_map: Dict[str, Any] = {
            "has_leading_vehicle":           checker.check_has_leading_vehicle,
            "thw_in_range":                  checker.check_thw_in_range,
            "ego_speed_sufficient":          checker.check_ego_speed_sufficient,
            "leader_no_decel_trend":         checker.check_leader_no_decel_trend,
            "has_adjacent_vehicle":          checker.check_has_adjacent_vehicle,
            "adjacent_lateral_oscillation":  checker.check_adjacent_lateral_oscillation,
            "adjacent_in_merging_zone":      checker.check_adjacent_in_merging_zone,
            "lateral_gap_available":         checker.check_lateral_gap_available,
            "near_intersection":             checker.check_near_intersection,
            "has_opposing_vehicle":          checker.check_has_opposing_vehicle,
            "has_third_conflicting_vehicle": checker.check_has_third_conflicting_vehicle,
            "complex_scene":                 checker.check_complex_scene,
        }

        met: List[str] = []
        failed: List[str] = []
        total_conf: float = 0.0
        detail_parts: List[str] = []

        for cond in preconditions:
            fn = check_map.get(cond)
            if fn is None:
                logger.warning(f"LongTailAssessor: unknown precondition '{cond}'")
                continue
            try:
                is_met, conf, detail = fn()
            except Exception as exc:
                logger.error(f"LongTailAssessor: check '{cond}' raised: {exc}")
                is_met, conf, detail = False, 0.0, f"error: {exc}"

            symbol = "✓" if is_met else "✗"
            detail_parts.append(f"{cond}={symbol}({detail})")

            if is_met:
                met.append(cond)
                total_conf += conf
            else:
                failed.append(cond)

        n = len(preconditions)
        if n == 0:
            return 0.0, [], [], ""

        frac_met = len(met) / n
        mean_conf = total_conf / len(met) if met else 0.0
        score = frac_met * mean_conf

        return score, met, failed, "; ".join(detail_parts)

    @staticmethod
    def _empty_result(reason: str = "") -> Dict[str, Any]:
        return {
            "longtail_subtype":     "",
            "potential_score":      0.0,
            "met_preconditions":    [],
            "failed_preconditions": [],
            "weight_modifiers":     {},
            "assessment_summary":   f"[LTA] 无有效长尾类型: {reason}",
        }
