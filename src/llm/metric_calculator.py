"""
MetricCalculator — 基于对抗策略投影轨迹的指标预估器

核心逻辑
--------
AnalysisAgent 输出的对抗策略包含：
  attacker_vehicle_id, timing_window, parameter_changes
  (speed_delta_per_step, heading_delta_per_step, intervention_steps)

本模块在 **原始预测轨迹** 基础上，将攻击参数叠加到攻击窗口内的轨迹点，
生成 **投影后的对抗轨迹**，再基于该轨迹计算 TTC / MinDist_lat / YawRate / THW。

这与训练阶段 AdvGenLoss.forward() 的指标计算逻辑对应：
  - MetricCalculator 在优化 **前** 预估指标（供 ReflectionAgent 设权重）
  - 训练时每步梯度迭代从 CVAE 生成的轨迹实时重算（权重固定，指标随优化进化）
"""

import json
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricCalculator:
    """
    从场景 JSON + AnalysisAgent 策略，预估对抗行为执行后的安全指标。

    使用方法
    --------
    calc = MetricCalculator(scenario_json_str_or_dict)
    result = calc.calculate_metrics(calculator_input)

    calculator_input 结构：
    {
        "driver_agent_inputs": {"priority_metrics": ["TTC", "MinDist_lat", "YawRate"]},
        "key_interaction": {
            "attacker_vehicle_id": "vehicle_1",
            "target_vehicle_id":   "ego_vehicle"
        },
        "adversarial_strategy": {           # 来自 AnalysisAgent step4 / selected_behavior
            "timing_window":   [1.0, 3.0],
            "parameter_changes": {
                "speed_delta_per_step":   1.5,
                "heading_delta_per_step": 7.0
            }
        }
    }
    """

    def __init__(self, scenario_data: Any) -> None:
        """
        Args:
            scenario_data: ScenarioExtractor 返回的场景描述。
                - str  : JSON 字符串
                - dict : 已解析的场景字典
                - 其他 : 不支持，输出警告并降级
        """
        self._vehicles: Dict[Any, Dict] = {}
        self._dt: float = 0.5
        self._loaded: bool = False

        if isinstance(scenario_data, str):
            try:
                self._scenario = json.loads(scenario_data)
                self._index_vehicles()
                self._loaded = True
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error(f"MetricCalculator: failed to parse scenario JSON: {exc}")
        elif isinstance(scenario_data, dict):
            self._scenario = scenario_data
            self._index_vehicles()
            self._loaded = True
        else:
            logger.warning("MetricCalculator: non-JSON data received, numerical calc unavailable")
            self._scenario = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _index_vehicles(self) -> None:
        self._dt = self._scenario.get("dt", 0.5)
        for v in self._scenario.get("vehicles", []):
            vid = v.get("id", 0)
            self._vehicles[vid] = v
            if v.get("is_ego"):
                self._vehicles["ego"] = v

    def _resolve_vehicle(self, vid_str: str) -> Optional[Dict]:
        if not vid_str:
            return None
        s = str(vid_str).lower()
        if "ego" in s:
            return self._vehicles.get("ego")
        digits = "".join(c for c in s if c.isdigit())
        if digits:
            idx = int(digits)
            v = self._vehicles.get(idx)
            # vehicle_0 is always ego — return ego entry for consistency
            if v is not None and v.get("is_ego"):
                return self._vehicles.get("ego", v)
            return v
        return None

    # ------------------------------------------------------------------
    # Adversarial trajectory projection
    # ------------------------------------------------------------------

    def _project_attacker_trajectory(
        self,
        attacker: Dict,
        strategy: Dict,
    ) -> List[Dict]:
        """
        在原始预测轨迹基础上叠加攻击参数，生成对抗投影轨迹。

        攻击窗口 [t_start, t_end] 内每个时间步：
          - 速度累加 speed_delta_per_step（不低于 0）
          - 航向累加 heading_delta_per_step（归一化到 [-180, 180]）
          - 位置按新速度/航向向前推算

        窗口外沿用原始轨迹点。

        Returns:
            投影后的未来轨迹（t >= 0），包含字段 projected=True/False
        """
        param_changes = strategy.get("parameter_changes", {})
        timing_window = strategy.get("timing_window", [0.5, 6.0])
        t_start = float(timing_window[0]) if isinstance(timing_window, (list, tuple)) else 0.5
        t_end   = float(timing_window[1]) if isinstance(timing_window, (list, tuple)) else 6.0

        speed_delta   = float(param_changes.get("speed_delta_per_step", 0.0))
        heading_delta = float(param_changes.get("heading_delta_per_step", 0.0))

        # Sort future trajectory points
        future = sorted(
            [pt for pt in attacker.get("trajectory", []) if pt.get("t", -99) >= 0],
            key=lambda p: p["t"],
        )
        if not future:
            logger.warning("MetricCalculator: attacker has no future trajectory points")
            return []

        projected: List[Dict] = []
        # Running state for integration
        curr_x       = future[0]["x"]
        curr_y       = future[0]["y"]
        curr_speed   = future[0].get("velocity", 0.0)
        curr_heading = future[0].get("heading", 0.0)
        in_attack    = False

        for i, pt in enumerate(future):
            t = pt["t"]

            if t < t_start:
                # Before attack — copy original point
                curr_x       = pt["x"]
                curr_y       = pt["y"]
                curr_speed   = pt.get("velocity", curr_speed)
                curr_heading = pt.get("heading", curr_heading)
                projected.append({**pt, "projected": False})
                continue

            if not in_attack and t >= t_start:
                in_attack = True
                # Anchor position from previous step
                if projected:
                    prev = projected[-1]
                    curr_x, curr_y = prev["x"], prev["y"]

            if t_start <= t <= t_end:
                # Apply adversarial delta
                curr_speed   = max(0.0, curr_speed + speed_delta)
                curr_heading += heading_delta
                while curr_heading >  180: curr_heading -= 360
                while curr_heading < -180: curr_heading += 360
            # else: t > t_end → maintain last modified state (coast at constant speed/heading)

            # Integrate position
            h_rad = math.radians(curr_heading)
            curr_x += curr_speed * math.cos(h_rad) * self._dt
            curr_y += curr_speed * math.sin(h_rad) * self._dt

            projected.append({
                "t":        t,
                "x":        round(curr_x, 4),
                "y":        round(curr_y, 4),
                "heading":  round(curr_heading, 4),
                "velocity": round(curr_speed, 4),
                "projected": True,
            })

        return projected

    # ------------------------------------------------------------------
    # Metric calculations on paired trajectories
    # ------------------------------------------------------------------

    @staticmethod
    def _vel_vec(pt: Dict) -> Tuple[float, float]:
        hr = math.radians(pt.get("heading", 0.0))
        v  = pt.get("velocity", 0.0)
        return v * math.cos(hr), v * math.sin(hr)

    def _paired_points(
        self, traj_a: List[Dict], traj_b: List[Dict]
    ) -> List[Tuple[Dict, Dict]]:
        """Match trajectory points by timestamp (t >= 0)."""
        t_b = {round(pt["t"], 2): pt for pt in traj_b if pt.get("t", -99) >= 0}
        return [
            (pa, t_b[round(pa["t"], 2)])
            for pa in traj_a
            if pa.get("t", -99) >= 0 and round(pa["t"], 2) in t_b
        ]

    def _calc_ttc(self, proj_attacker: List[Dict], ego_traj: List[Dict]) -> Dict:
        """TTC along ego longitudinal axis."""
        results = []
        for a_pt, e_pt in self._paired_points(proj_attacker, ego_traj):
            dx = a_pt["x"] - e_pt["x"]
            dy = a_pt["y"] - e_pt["y"]
            h_rad = math.radians(e_pt.get("heading", 0.0))
            cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
            d_long = dx * cos_h + dy * sin_h

            a_vx, a_vy = self._vel_vec(a_pt)
            e_vx, e_vy = self._vel_vec(e_pt)
            v_rel = (a_vx - e_vx) * cos_h + (a_vy - e_vy) * sin_h

            if d_long > 0 and v_rel < 0:
                ttc = min(-d_long / v_rel, 99.0)
            elif d_long < 0 and v_rel > 0:
                ttc = min(d_long / v_rel, 99.0)
            else:
                ttc = 99.0

            results.append({
                "t": a_pt["t"],
                "d_long": round(d_long, 3),
                "v_rel":  round(v_rel, 3),
                "ttc":    round(max(0.0, ttc), 3),
            })

        if not results:
            return {"min_value": 99.0, "values": [], "risk_level": "low"}
        best = min(results, key=lambda r: r["ttc"])
        rl = "high" if best["ttc"] < 2.0 else "medium" if best["ttc"] < 3.0 else "low"
        return {"min_value": best["ttc"], "min_time": best["t"], "values": results, "risk_level": rl}

    def _calc_min_dist_lat(
        self, proj_attacker: List[Dict], ego_traj: List[Dict],
        ego_width: float, att_width: float
    ) -> Dict:
        """Minimum lateral gap (vehicle-edge to vehicle-edge)."""
        results = []
        for a_pt, e_pt in self._paired_points(proj_attacker, ego_traj):
            dx = a_pt["x"] - e_pt["x"]
            dy = a_pt["y"] - e_pt["y"]
            h_rad = math.radians(e_pt.get("heading", 0.0))
            perp_x, perp_y = -math.sin(h_rad), math.cos(h_rad)
            d_lat = abs(dx * perp_x + dy * perp_y)
            gap   = d_lat - (ego_width + att_width) / 2.0
            results.append({"t": a_pt["t"], "d_lat": round(d_lat, 3), "gap": round(gap, 3)})

        if not results:
            return {"min_value": 99.0, "values": [], "risk_level": "low"}
        best = min(results, key=lambda r: r["gap"])
        rl = "high" if best["gap"] < 0.5 else "medium" if best["gap"] < 1.5 else "low"
        return {"min_value": best["gap"], "min_time": best["t"], "values": results, "risk_level": rl}

    def _calc_yaw_rate(self, proj_attacker: List[Dict]) -> Dict:
        """Maximum absolute yaw rate of the projected attacker trajectory (deg/s)."""
        traj = sorted(
            [pt for pt in proj_attacker if pt.get("t", -99) >= 0],
            key=lambda p: p["t"],
        )
        results = []
        for i in range(len(traj) - 1):
            dh = traj[i + 1]["heading"] - traj[i]["heading"]
            while dh >  180: dh -= 360
            while dh < -180: dh += 360
            yr = abs(dh) / self._dt
            results.append({"t": traj[i]["t"], "yaw_rate": round(yr, 3)})

        if not results:
            return {"max_value": 0.0, "values": [], "risk_level": "low"}
        best = max(results, key=lambda r: r["yaw_rate"])
        rl = "high" if best["yaw_rate"] > 15 else "medium" if best["yaw_rate"] > 8 else "low"
        return {"max_value": best["yaw_rate"], "max_time": best["t"], "values": results, "risk_level": rl}

    def _calc_thw(self, proj_attacker: List[Dict], ego_traj: List[Dict]) -> Dict:
        """THW = longitudinal distance / ego speed (only when attacker is ahead)."""
        results = []
        for a_pt, e_pt in self._paired_points(proj_attacker, ego_traj):
            dx = a_pt["x"] - e_pt["x"]
            dy = a_pt["y"] - e_pt["y"]
            h_rad = math.radians(e_pt.get("heading", 0.0))
            cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
            d_long = dx * cos_h + dy * sin_h
            if d_long <= 0:
                continue
            v_ego = max(e_pt.get("velocity", 0.01), 0.01)
            thw = min(d_long / v_ego, 99.0)
            results.append({"t": a_pt["t"], "d_long": round(d_long, 3), "thw": round(thw, 3)})

        if not results:
            return {"min_value": 99.0, "values": [], "risk_level": "low"}
        best = min(results, key=lambda r: r["thw"])
        rl = "high" if best["thw"] < 1.0 else "medium" if best["thw"] < 2.0 else "low"
        return {"min_value": best["thw"], "min_time": best["t"], "values": results, "risk_level": rl}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_metrics(self, calculator_input: Dict) -> Dict:
        """
        主入口：根据 AnalysisAgent 输出的对抗策略，投影攻击轨迹并计算预期指标。

        Args:
            calculator_input: 包含以下字段的字典
                driver_agent_inputs.priority_metrics : 要计算的指标列表
                key_interaction.attacker_vehicle_id  : 攻击车辆 ID
                key_interaction.target_vehicle_id    : 目标车辆 ID (通常是 ego)
                adversarial_strategy                 : 来自 AnalysisAgent 的策略参数
                  .timing_window       : [t_start, t_end]
                  .parameter_changes   :
                    .speed_delta_per_step   : 每步速度增量 (m/s)
                    .heading_delta_per_step : 每步航向增量 (deg)

        Returns:
            与 ReflectionAgent 输入格式兼容的指标报告
        """
        if not self._loaded:
            logger.error("MetricCalculator: not initialized with valid scenario data")
            return {}

        metrics_list: List[str] = (
            calculator_input.get("driver_agent_inputs", {}).get("priority_metrics", [])
            or ["TTC", "MinDist_lat", "YawRate"]
        )
        attacker_id: str = calculator_input.get("key_interaction", {}).get("attacker_vehicle_id", "")
        target_id:   str = calculator_input.get("key_interaction", {}).get("target_vehicle_id", "ego_vehicle")
        strategy:    Dict = calculator_input.get("adversarial_strategy", {})

        attacker = self._resolve_vehicle(attacker_id)
        target   = self._resolve_vehicle(target_id)

        if attacker is None or target is None:
            logger.error(
                f"MetricCalculator: cannot resolve vehicles "
                f"(attacker='{attacker_id}', target='{target_id}'). "
                f"Available ids: {list(self._vehicles.keys())}"
            )
            return {}

        # ── 1. Project adversarial trajectory ──────────────────────────
        proj_traj = self._project_attacker_trajectory(attacker, strategy)
        if not proj_traj:
            logger.warning("MetricCalculator: projection failed, falling back to original trajectory")
            proj_traj = [
                pt for pt in attacker.get("trajectory", []) if pt.get("t", -99) >= 0
            ]

        ego_future = [pt for pt in target.get("trajectory", []) if pt.get("t", -99) >= 0]

        logger.info(
            f"MetricCalculator: projected {len(proj_traj)} future points for {attacker_id} "
            f"(strategy={strategy.get('parameter_changes', {})})"
        )

        # ── 2. Compute requested metrics ────────────────────────────────
        computed: Dict[str, Any] = {}
        for metric in metrics_list:
            m = metric.strip()
            if m == "TTC":
                computed["TTC"] = self._calc_ttc(proj_traj, ego_future)
            elif m == "MinDist_lat":
                computed["MinDist_lat"] = self._calc_min_dist_lat(
                    proj_traj, ego_future,
                    target.get("width", 1.73),
                    attacker.get("width", 1.73),
                )
            elif m == "YawRate":
                computed["YawRate"] = self._calc_yaw_rate(proj_traj)
            elif m == "THW":
                computed["THW"] = self._calc_thw(proj_traj, ego_future)
            else:
                logger.warning(f"MetricCalculator: '{m}' not implemented, skipped")

        # ── 3. Aggregate risk ───────────────────────────────────────────
        risk_levels = [v.get("risk_level", "low") for v in computed.values()]
        overall = (
            "high"   if "high"   in risk_levels else
            "medium" if "medium" in risk_levels else
            "low"
        )

        summary_parts = []
        for name, data in computed.items():
            if "min_value" in data:
                summary_parts.append(
                    f"{name}={data['min_value']:.2f}s/{data['min_value']:.2f}m "
                    f"({data['risk_level']})"
                )
            elif "max_value" in data:
                summary_parts.append(
                    f"{name}={data['max_value']:.1f}°/s ({data['risk_level']})"
                )

        timing_info = (
            f"attack window t={strategy.get('timing_window', 'unknown')}, "
            f"Δspeed={strategy.get('parameter_changes', {}).get('speed_delta_per_step', 0):.1f}m/s/step, "
            f"Δheading={strategy.get('parameter_changes', {}).get('heading_delta_per_step', 0):.1f}°/step"
        )

        return {
            "calculation_source": "numerical_projected",   # 区别于原始轨迹计算
            "attacker_id":   attacker_id,
            "target_id":     target_id,
            "strategy_applied": timing_info,
            "risk_assessment": overall,
            "metrics":  computed,
            "calculation_summary": (
                f"Projected adversarial trajectory ({timing_info}). "
                + ", ".join(summary_parts)
            ),
            "weight_adjustment_guidance": (
                "Use projected metric values for ReflectionAgent dynamic-adjustment rules. "
                "Note: actual values will evolve during STRIVE optimization."
            ),
        }
