"""
ReflectionAgent — 基于 AnalysisAgent 策略 + MetricCalculator/DriverAgent 数值，
调用 LLM 生成最终 AdvGenLoss 损失权重。

流程：
  1. 从 behavior_analysis 中提取行为标签、碰撞类型、优先级顺序
  2. 从 risk_metrics 中提取关键指标值（TTC、MinDist_lat、YawRate 等）
  3. 将以上信息及 loss_functions_kb 知识注入 Prompt
  4. 调用 LLM，获得 reasoning + risk_weights
  5. 解析 JSON，返回给 WeightManager
"""

import json
import logging
import textwrap
from typing import Any, Dict, Optional

from rich import print
from langchain_core.messages import HumanMessage, SystemMessage

from longterm.core.llm_factory import BaseAgent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Loss function key metadata (inline, so no extra file read needed)
# ──────────────────────────────────────────────────────────────────────

_LOSS_FUNCTIONS_SUMMARY = """
## STRIVE AdvGenLoss 损失函数权重速查表

| 权重键 | 类别 | 作用 | 有效范围 | 说明 |
|--------|------|------|----------|------|
| adv_crash | 攻击 | 最小化攻击车与 ego 的距离，核心碰撞驱动项，**始终最高** | 3.0–10.0 | 输出量级：软最小加权距离平方 |
| ttc | 攻击 | 最小化碰撞时间（TTC），切入/急刹/闯灯时调高 | 2.0–8.0 | 输出量级：0~8 秒原始值；**仅当攻击者比 ego 慢（接近中）才有梯度** |
| thw | 攻击 | Time Headway：维持攻击车在 ego 正前方的近距跟随，**与 TTC 互补** | 1.0–4.0 | 输出量级：0~2.5 秒；速度相近时 TTC=∞ 无梯度，THW 仍有效；**纵向急刹场景专用** |
| min_dist_lat | 攻击 | 最小化横向间距，横向挤压/变道时调高 | 1.0–6.0 | 输出量级：k×gap²，k=2，约 0~50 |
| yaw_rate | 攻击 | 鼓励超过舒适横摆阈值(15°/s)，转向场景调高 | **15.0–60.0** | ⚠️ 输出量级极小(0~0.26 rad/s)，必须用高权重才有效梯度 |
| yaw_rate_ego | 攻击 | ego 横摆贡献，ego 须急转规避时调高 | **5.0–30.0** | 纵向碰撞场景(急刹)调低至 5 |
| yaw_rate_non_ego | 攻击 | 攻击车横摆贡献，攻击车须急转时调高 | **20.0–60.0** | 切入/路口等需要大转向场景调高 |
| coll_veh | 约束 | 防止背景车辆互相碰撞，通常保持激活 | 5.0–20.0 | 输出量级：归一化 0~1 |
| coll_veh_plan | 约束 | 防止 ego 撞上非目标车辆，通常保持激活 | 5.0–20.0 | 同上 |
| coll_env | 约束 | 防止车辆驶出可行驶区域，LaneDeparture 可适当放松 | 5.0–20.0 | 同上 |
| motion_prior_atk | 约束 | 攻击车轨迹真实性，**必须放松（调低）**以允许激进行为，但不可过低 | **0.0005–0.003** | 过低(<0.0003)会产生非物理轨迹，保持合理范围确保场景真实性 |
| init_z_atk | 约束 | 攻击车潜变量稳定性，随 motion_prior_atk 一起放松 | 0.005–0.05 | |

**核心决策逻辑**：
- adv_crash 始终最高（碰撞驱动核心）
- **thw 与 ttc 互补**：ttc 仅在速度差存在时有效；thw 在同速跟随阶段提供持续梯度，优先在 SuddenBraking 系列场景中配合 ttc 同时使用
- motion_prior_atk 和 init_z_atk 调低以允许激进行为，但 motion_prior_atk 不低于 0.0005（保证轨迹物理合理性）
- yaw_rate 系列：因损失函数输出量级极小（0~0.26 rad/s），**必须设置较大权重（15~60）才能有效影响优化梯度**；设置 1~5 等小值实际上没有任何效果
- ttc : thw : min_dist_lat : yaw_rate 的比例决定攻击类型（时机/建立前提 vs 横向 vs 转向）
- risk_level=high_risk → 攻击项全力拉满；longtail_condition → 适度；low_risk → 保守
"""

_BEHAVIOR_WEIGHT_HINTS = """
## 按行为类型的权重调整指引

⚠️ 关键提醒一：yaw_rate 系列的损失函数输出量级极小（0~0.26 rad/s），设置小值（<10）没有任何实际梯度效果。
必须设置 15~60 才能与 ttc/adv_crash 量级匹配，使转向行为真正被优化驱动。

⚠️ 关键提醒二：thw 与 ttc 互补，针对纵向场景尤其重要。
- **纵向攻击**（前车制动类）：ttc 仅当相对接近速度出现后才有梯度；优化初期双方同速时 TTC=∞，
  此时 thw 提供持续梯度维持近距跟随；建议 ttc 与 thw 同时调高。
- **横向/路口攻击**（切入/转弯类）：攻击者多来自侧方，不在 ego 前方，thw 基本无效；可设 0.5 或禁用。

| 行为 | thw 建议值 | yaw_rate_non_ego 建议值 | yaw_rate_ego 建议值 | 其他重点 |
|------|-----------|------------------------|--------------------|---------| 
| AggressiveCutIn | **0.5**（攻击者侧方，不在前方） | **30–50**（急速变道） | 8–15（ego轻微规避） | adv_crash↑, min_dist_lat↑ |
| SuddenBraking | **2.0–3.0**（核心前提条件） | **8–15**（基本无转向） | **5–8**（纵向场景） | adv_crash↑↑, ttc↑↑, thw↑↑ |
| IntersectionRush | **1.0**（部分场景攻击者先在前方） | **30–50**（路口转弯） | 12–20（ego转弯规避） | adv_crash↑↑, ttc↑ |
| LaneDeparture | **0.5**（侧方场景） | **35–55**（极高偏摆） | **25–40**（ego侧偏） | adv_crash↑, min_dist_lat↑ |
| StationaryObstacleActivation | **1.5**（静止障碍物在前方） | **15–25**（中等转向） | 5–10 | adv_crash↑, ttc↑ |
| MultiVehiclePincer | **1.0**（包抄中有车在前） | **30–50**（多方向包抄） | 12–20 | adv_crash↑↑, ttc↑ |
| SuddenAcceleration | **0.5**（加速离开，不是近距在前） | **10–20**（加速微转向） | 5–10 | adv_crash↑, ttc↑ |

## 长尾子类型额外指引

| 长尾子类型 | 场景特征 | 关键调整方向 |
|------------|----------|-------------|
| SuddenBrakingUnpredictable | 前车无减速迹象时突然急刹 | **thw 取上限(2.5–3.5)** + ttc 取上限，adv_crash↑，yaw_rate_non_ego 取下限(8~15)，motion_prior_atk 取 0.001~0.002 |
| HesitantCutIn | 邻车摆动后突然并线 | thw=0.5（攻击者侧方），min_dist_lat↑，yaw_rate_non_ego 取上限(40~55)，adv_crash↑ |
| OccludedConflict | 对向车遮挡下的路口冲突 | thw=1.0（路口场景），adv_crash 取上限，ttc↑，yaw_rate_non_ego 30~45，coll_veh 可小幅降低 |

motion_prior_atk 在任何场景均需保持在 0.0005~0.003 之间，**不可设为 0 或极小值**，否则攻击轨迹将失去物理合理性。
"""

# ──────────────────────────────────────────────────────────────────────
# 确定性回退（当 LLM 调用失败时使用）
# ──────────────────────────────────────────────────────────────────────

_FALLBACK_TEMPLATES: Dict[str, Dict[str, float]] = {
    # yaw_rate 系列已按 YawRateLoss 实际输出量级（0~0.26 rad/s）重新标定：
    # 设置 20~50 才能与 ttc/adv_crash 的梯度量级匹配。
    # thw 已按场景属性标定：纵向前方跟随场景调高(2.0–3.0)，侧方/横向场景调低(0.5)。
    #
    # AggressiveCutIn：高速切入，攻击车来自侧方，thw 弱（不在前方）
    "AggressiveCutIn":              {"adv_crash": 6.0, "ttc": 2.5, "thw": 0.5,  "min_dist_lat": 4.0, "yaw_rate": 25.0, "yaw_rate_ego": 10.0, "yaw_rate_non_ego": 38.0, "coll_veh": 10.0, "coll_veh_plan": 10.0, "coll_env": 12.0, "motion_prior_atk": 0.001, "init_z_atk": 0.01},
    # SuddenBraking：纵向急刹，攻击车在前方，thw 是最重要的前提条件损失
    "SuddenBraking":                {"adv_crash": 5.0, "ttc": 5.0, "thw": 2.5,  "min_dist_lat": 1.0, "yaw_rate": 15.0, "yaw_rate_ego": 6.0,  "yaw_rate_non_ego": 12.0, "coll_veh": 12.0, "coll_veh_plan": 12.0, "coll_env": 15.0, "motion_prior_atk": 0.002, "init_z_atk": 0.02},
    # IntersectionRush：路口高速转弯冲突，攻击者从侧向切入，thw 中等
    "IntersectionRush":             {"adv_crash": 7.0, "ttc": 4.0, "thw": 1.0,  "min_dist_lat": 3.0, "yaw_rate": 28.0, "yaw_rate_ego": 15.0, "yaw_rate_non_ego": 40.0, "coll_veh": 10.0, "coll_veh_plan": 10.0, "coll_env": 12.0, "motion_prior_atk": 0.001, "init_z_atk": 0.01},
    # LaneDeparture：最强横摆需求，侧方场景，thw 弱
    "LaneDeparture":                {"adv_crash": 5.0, "ttc": 2.0, "thw": 0.5,  "min_dist_lat": 4.0, "yaw_rate": 38.0, "yaw_rate_ego": 28.0, "yaw_rate_non_ego": 48.0, "coll_veh": 10.0, "coll_veh_plan": 10.0, "coll_env": 8.0,  "motion_prior_atk": 0.001, "init_z_atk": 0.01},
    # StationaryObstacleActivation：静止障碍物在前方激活，thw 中等
    "StationaryObstacleActivation": {"adv_crash": 6.0, "ttc": 3.5, "thw": 1.5,  "min_dist_lat": 3.0, "yaw_rate": 18.0, "yaw_rate_ego": 6.0,  "yaw_rate_non_ego": 22.0, "coll_veh": 10.0, "coll_veh_plan": 10.0, "coll_env": 12.0, "motion_prior_atk": 0.0007, "init_z_atk": 0.007},
    # MultiVehiclePincer：多方向包抄，包含前方车辆，thw 中等
    "MultiVehiclePincer":           {"adv_crash": 7.0, "ttc": 3.5, "thw": 1.0,  "min_dist_lat": 3.5, "yaw_rate": 30.0, "yaw_rate_ego": 14.0, "yaw_rate_non_ego": 42.0, "coll_veh": 8.0,  "coll_veh_plan": 10.0, "coll_env": 12.0, "motion_prior_atk": 0.001, "init_z_atk": 0.01},
    # SuddenAcceleration：攻击者加速离开，不在前方近距，thw 弱
    "SuddenAcceleration":           {"adv_crash": 5.5, "ttc": 4.0, "thw": 0.5,  "min_dist_lat": 2.0, "yaw_rate": 17.0, "yaw_rate_ego": 7.0,  "yaw_rate_non_ego": 20.0, "coll_veh": 12.0, "coll_veh_plan": 12.0, "coll_env": 15.0, "motion_prior_atk": 0.001, "init_z_atk": 0.01},
}

_DEFAULT_FALLBACK = {
    "adv_crash": 5.0, "ttc": 3.0, "thw": 1.0, "min_dist_lat": 2.5,
    "yaw_rate": 20.0, "yaw_rate_ego": 8.0, "yaw_rate_non_ego": 25.0,
    "coll_veh": 12.0, "coll_veh_plan": 12.0, "coll_env": 12.0,
    "motion_prior_atk": 0.002, "init_z_atk": 0.02,
}


def _extract_behavior_info(behavior_analysis: Any) -> Dict[str, Any]:
    """从 behavior_analysis（dict 或含 full_content 的包装）中提取关键字段。"""
    ba = behavior_analysis
    if isinstance(ba, dict) and "full_content" in ba:
        raw = ba["full_content"]
        if isinstance(raw, str):
            try:
                ba = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                ba = {}

    if not isinstance(ba, dict):
        return {}

    sb = ba.get("selected_behavior", {})
    ai = ba.get("agent_instructions", {})
    ri = ai.get("reflection_agent_inputs", {})
    cot = ba.get("cot_reasoning", {})

    return {
        "behavior_label": sb.get("behavior_label", ""),
        "collision_type": sb.get("collision_type", ""),
        "attacker_vehicle_id": sb.get("attacker_vehicle_id", ""),
        "priority_order": ri.get("loss_priority_order", []),
        "constraints_to_relax": ri.get("constraints_to_relax", []),
        "cot_summary": str(cot.get("step4_behavior_selection", ""))[:300],
    }


def _extract_metric_summary(risk_metrics: Optional[Dict]) -> str:
    """将 risk_metrics 压缩为简洁的文字摘要，便于注入 Prompt。"""
    if not risk_metrics or not isinstance(risk_metrics, dict):
        return "（无指标数据）"

    metrics = risk_metrics.get("metrics", risk_metrics)
    lines = []
    for name, data in metrics.items():
        if not isinstance(data, dict):
            continue
        if "min_value" in data:
            lines.append(
                f"  {name}: min={data['min_value']:.2f}  "
                f"风险={data.get('risk_level', '?')}  "
                f"({data.get('risk_explanation', '')})"
            )
        elif "max_value" in data:
            lines.append(
                f"  {name}: max={data['max_value']:.2f}  "
                f"风险={data.get('risk_level', '?')}  "
                f"({data.get('risk_explanation', '')})"
            )

    # DriverAgent 的 recommendations
    rec = risk_metrics.get("recommendations_for_reflection_agent", {})
    if rec:
        lines.append(f"\n  DriverAgent建议调高: {rec.get('increase', [])}")
        lines.append(f"  DriverAgent建议调低: {rec.get('decrease', [])}")
        lines.append(f"  理由: {rec.get('reasoning', '')}")

    return "\n".join(lines) if lines else "（指标数据为空）"


class ReflectionAgent(BaseAgent):
    """
    基于 LLM 的权重生成器。

    输入:  AnalysisAgent 的 behavior_analysis + MetricCalculator/DriverAgent 的 risk_metrics
    输出:  包含 reasoning 和 risk_weights 的 JSON 字符串，供 WeightManager 使用
    """

    def __init__(
        self,
        temperature: float = 0,
        verbose: bool = False,
        provider: Optional[str] = None,
    ) -> None:
        super().__init__(temperature=temperature, verbose=verbose, provider=provider)
        self.verbose = verbose

    def reflect_and_generate_weights(
        self,
        behavior_analysis: Dict[str, Any],
        risk_metrics: Optional[Dict[str, Any]] = None,
        risk_level: str = "high_risk",
        longtail_assessment: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调用 LLM，根据场景分析结果和风险指标生成最优损失权重。

        Returns:
            JSON 字符串，格式: {"reasoning": "...", "risk_weights": {...}}
        """
        print("[cyan]ReflectionAgent: 调用 LLM 生成损失权重...[/cyan]")

        # ── 1. 提取上游输出的关键信息 ──────────────────────────────────
        binfo = _extract_behavior_info(behavior_analysis)
        behavior_label   = binfo.get("behavior_label", "unknown")
        collision_type   = binfo.get("collision_type", "")
        attacker_id      = binfo.get("attacker_vehicle_id", "")
        priority_order   = binfo.get("priority_order", [])
        relax_list       = binfo.get("constraints_to_relax", [])
        cot_summary      = binfo.get("cot_summary", "")
        metric_summary   = _extract_metric_summary(risk_metrics)

        # ── 2. 构建 System Prompt ──────────────────────────────────────
        system_message = textwrap.dedent(f"""\
        你是 STRIVE 对抗场景生成系统的 **权重决策专家（ReflectionAgent）**。
        你的任务是根据前两个 Agent 的分析结果，为 AdvGenLoss 优化器生成最优损失函数权重。

        {_LOSS_FUNCTIONS_SUMMARY}

        {_BEHAVIOR_WEIGHT_HINTS}

        ## 输出格式（严格遵守）
        只输出以下 JSON，不添加任何额外说明：
        ```json
        {{
          "reasoning": "简洁说明每个关键权重的取值原因（2-4句话）",
          "risk_weights": {{
            "adv_crash":          <float>,
            "ttc":                <float>,
            "thw":                <float>,
            "min_dist_lat":       <float>,
            "yaw_rate":           <float>,
            "yaw_rate_ego":       <float>,
            "yaw_rate_non_ego":   <float>,
            "coll_veh":           <float>,
            "coll_veh_plan":      <float>,
            "coll_env":           <float>,
            "motion_prior_atk":   <float>,
            "init_z_atk":         <float>
          }}
        }}
        ```

        ## 硬性约束
        1. risk_weights 中必须包含且仅包含以上 12 个键，不增不减
        2. adv_crash 必须是攻击类权重中最大的值
        3. motion_prior_atk ∈ [0.0005, 0.003]，init_z_atk ∈ [0.005, 0.05]（必须放松，但不可过低以保证轨迹真实性）
        4. yaw_rate ∈ [15.0, 60.0]，yaw_rate_ego ∈ [5.0, 30.0]，yaw_rate_non_ego ∈ [20.0, 60.0]（小值无效，必须在此范围内）
        5. thw ∈ [0.0, 4.0]；纵向急刹场景（SuddenBraking 系列）建议 2.0–3.5；横向/路口场景建议 0.5–1.0
        6. 所有权重必须为正数
        7. 不输出任何 Markdown 格式以外的额外文字
        """)

        # ── 3. 构建 Human Prompt（含长尾评估注入） ────────────────────
        lta_section = ""
        if longtail_assessment and longtail_assessment.get("longtail_subtype"):
            lta_subtype = longtail_assessment.get("longtail_subtype", "")
            lta_score   = longtail_assessment.get("potential_score", 0.0)
            lta_met     = longtail_assessment.get("met_preconditions", [])
            lta_summary = longtail_assessment.get("assessment_summary", "")
            lta_section = textwrap.dedent(f"""\

            ## 长尾潜力评估结果（LongTailPotentialAssessor 输出）

            **长尾子类型**: {lta_subtype}  （潜力评分: {lta_score:.2f}/1.0）
            **满足前提条件**: {', '.join(lta_met) if lta_met else '无'}
            **评估摘要**: {lta_summary}

            > 提示：该场景被判定具备生成 **{lta_subtype}** 长尾事件的潜力。
            > 请参考上方"长尾子类型额外指引"表格，针对性地调整权重比例，使生成的对抗行为更贴合该长尾模式。
            > ⚠️ 注意：权重仍须满足所有硬性约束（adv_crash 最大，motion_prior_atk 极低等）。
            """)

        human_message = textwrap.dedent(f"""\
        ## 当前场景分析结果

        **目标风险等级**: {risk_level}
        **行为标签**: {behavior_label}
        **碰撞类型**: {collision_type}
        **攻击车辆**: {attacker_id}
        **损失优先级（AnalysisAgent 指定）**: {' > '.join(priority_order) if priority_order else '未指定'}
        **建议放松的约束项**: {', '.join(relax_list) if relax_list else '未指定'}
        **AnalysisAgent 行为选择摘要**: {cot_summary}

        ## MetricCalculator/DriverAgent 计算的风险指标

        {metric_summary}
        {lta_section}
        ## 你的任务

        综合以上信息，为 AdvGenLoss 优化器生成最优损失权重。
        要求：
        1. 根据行为类型（{behavior_label}）和风险等级（{risk_level}）确定各攻击项的比例
        2. 根据指标值判断哪些方向的推力不足（指标安全 → 对应权重调高），哪些方向已足够（指标危险 → 无需额外推力）
        3. 遵守 AnalysisAgent 的优先级排序建议（如有）
        4. 若存在长尾评估结果，参照对应子类型的权重调整指引
        5. 输出符合格式要求的 JSON

        请直接输出 JSON，不要有任何前缀说明。
        """)

        # ── 4. 调用 LLM ───────────────────────────────────────────────
        import time as _time
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        response_content = ""
        _printed_len = 0
        _t_api_start = _time.time()
        _first_token_time = None

        def _print_new(text: str) -> None:
            nonlocal _printed_len
            if len(text) > _printed_len:
                print(text[_printed_len:], end="", flush=True)
                _printed_len = len(text)

        try:
            for chunk in self.llm.stream(messages):
                if _first_token_time is None:
                    _first_token_time = _time.time() - _t_api_start
                piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                if isinstance(piece, str) and piece:
                    if piece.startswith(response_content) and len(piece) > len(response_content):
                        response_content = piece
                    else:
                        response_content += piece
                    _print_new(response_content)
        except Exception as e:
            print(f"[red]LLM 流式调用失败: {e}，尝试 invoke...[/red]")
            try:
                resp = self.llm.invoke(messages)
                response_content = resp.content if hasattr(resp, "content") else str(resp)
                print(response_content)
            except Exception as invoke_err:
                print(f"[red]LLM invoke 也失败: {invoke_err}，回退到确定性规则[/red]")
                return self._fallback(behavior_label, risk_level)

        _api_elapsed = _time.time() - _t_api_start
        self.last_call_duration = _api_elapsed
        self.last_first_token_time = _first_token_time or _api_elapsed
        print(f"\n[dim]  ⏱ API call: {_api_elapsed:.1f}s (first token: {self.last_first_token_time:.1f}s)[/dim]")

        # ── 5. 解析 JSON ───────────────────────────────────────────────
        import re
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_content, re.DOTALL)
        json_str = json_match.group(1) if json_match else response_content.strip()

        try:
            result = json.loads(json_str)
            weights = result.get("risk_weights", {})
            if not weights or len(weights) < 5:
                raise ValueError(f"risk_weights 字段为空或不完整: {weights}")

            print(f"[green]✓ LLM 生成 {len(weights)} 个权重[/green]")
            for k, v in sorted(weights.items(), key=lambda x: -float(x[1])):
                print(f"    {k:40s} = {float(v):8.3f}")

            # ── 长尾潜力修正（乘性叠加，确定性规则） ──────────────────
            weights, lta_log = self._apply_longtail_modifiers(weights, longtail_assessment)
            if lta_log:
                result["risk_weights"] = weights
                old_reasoning = result.get("reasoning", "")
                result["reasoning"] = old_reasoning + f" | LTA({lta_log['subtype']} score={lta_log['score']:.2f}): {lta_log['summary']}"

            return json.dumps(result, ensure_ascii=False, indent=2)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[red]JSON 解析失败: {e}，回退到确定性规则[/red]")
            return self._fallback(behavior_label, risk_level, longtail_assessment)

    # 硬下限：防止任何修正系数将真实性约束项压至无效水平
    _FLOOR_LIMITS: Dict[str, float] = {
        "motion_prior_atk": 0.0004,  # 低于此值轨迹会失去物理合理性
        "init_z_atk":       0.004,   # 与 motion_prior_atk 联动
        "coll_veh":         4.0,     # 防止多车碰撞约束完全失效
        "coll_veh_plan":    4.0,
        "coll_env":         3.0,
    }

    @staticmethod
    def _apply_longtail_modifiers(
        weights: Dict[str, float],
        longtail_assessment: Optional[Dict[str, Any]],
        min_score: float = 0.5,
    ):
        """
        对 LLM 生成的权重应用长尾潜力修正（乘性系数）。
        修正后对关键真实性约束项（motion_prior_atk 等）应用硬下限，
        确保攻击轨迹不会因过度放松约束而失去物理合理性。

        Returns:
            (modified_weights, log_dict_or_None)
            log_dict 结构: {"subtype": str, "score": float, "summary": str}
        """
        if not longtail_assessment or not isinstance(longtail_assessment, dict):
            return weights, None

        subtype  = longtail_assessment.get("longtail_subtype", "")
        score    = longtail_assessment.get("potential_score", 0.0)
        mods     = longtail_assessment.get("weight_modifiers", {})

        if not subtype or score < min_score or not mods:
            if subtype:
                print(f"[yellow]  LTA: {subtype} score={score:.2f} < {min_score}, 跳过修正[/yellow]")
            return weights, None

        print(f"\n[bold magenta]── Step 3.5: Long-Tail Weight Modifiers ──[/bold magenta]")
        print(f"[magenta]  子类型: {subtype}  |  潜力评分: {score:.2f}[/magenta]")

        floor = ReflectionAgent._FLOOR_LIMITS
        w = dict(weights)
        change_parts = []
        clamped_parts = []

        for k, factor in mods.items():
            if k in w:
                old_v = w[k]
                new_v = round(old_v * factor, 4)

                # 硬下限保护
                if k in floor and new_v < floor[k]:
                    new_v = floor[k]
                    clamped_parts.append(f"{k}→floor({floor[k]})")

                w[k] = new_v
                arrow = "↑" if factor > 1.0 else "↓"
                change_parts.append(f"{k}:{old_v:.4f}×{factor}{arrow}{w[k]:.4f}")
                print(f"    {k:25s}  {old_v:8.4f}  ×{factor:<5}  → {w[k]:8.4f}  {arrow}"
                      + (f"  [clamped]" if k in clamped_parts else ""))

        if clamped_parts:
            print(f"[yellow]  ⚠ 硬下限保护触发: {', '.join(clamped_parts)}（确保轨迹真实性）[/yellow]")

        summary = "; ".join(change_parts)
        print(f"[green]✓ 长尾修正完成 ({len(change_parts)} 项)[/green]")
        return w, {"subtype": subtype, "score": score, "summary": summary}

    def _fallback(
        self,
        behavior_label: str,
        risk_level: str,
        longtail_assessment: Optional[Dict[str, Any]] = None,
    ) -> str:
        """LLM 失败时的确定性兜底，仅用于异常恢复。"""
        print("[yellow]⚠ 使用确定性兜底权重（LLM 不可用）[/yellow]")
        template = _FALLBACK_TEMPLATES.get(behavior_label, _DEFAULT_FALLBACK)
        scale = {"high_risk": 1.0, "longtail_condition": 0.7, "low_risk": 0.3}.get(risk_level, 1.0)
        weights = {}
        attack_keys = {"adv_crash", "ttc", "thw", "min_dist_lat", "yaw_rate", "yaw_rate_ego", "yaw_rate_non_ego"}
        for k, v in template.items():
            if k in attack_keys:
                weights[k] = round(v * scale, 3)
            else:
                inv_scale = 1.0 / scale if scale < 1.0 else 1.0
                weights[k] = round(v * inv_scale, 4)

        weights, lta_log = self._apply_longtail_modifiers(weights, longtail_assessment)
        lta_note = f" | LTA({lta_log['subtype']})" if lta_log else ""
        result = {
            "reasoning": f"LLM 不可用，使用 {behavior_label} 模板 × {risk_level} 兜底{lta_note}",
            "risk_weights": weights,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
