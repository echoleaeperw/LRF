import os
import json
import re
from rich import print


def _parse_numeric(s) -> float:
    """Parse numeric value from strings like '+0.8 m/s', '-25 deg', 1.5, etc."""
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"[+-]?\d+\.?\d*", str(s))
    return float(m.group()) if m else 0.0


def _parse_timing_window(s) -> list:
    """
    Parse timing window from various formats:
      [1.0, 3.5], 't ∈ [1.0, 3.5]s', 't=[1.0, 3.5]', etc.
    """
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return [float(s[0]), float(s[1])]
    nums = re.findall(r"\d+\.?\d*", str(s))
    if len(nums) >= 2:
        return [float(nums[0]), float(nums[1])]
    return [0.5, 3.0]
from typing import Dict, Optional
from longterm.agents.analysis import AnalysisAgent
from longterm.agents.driver import DriverAgent
from longterm.agents.reflection import ReflectionAgent
from longterm.agents.longtail_assessor import LongTailPotentialAssessor
from longterm.core.content_processor import ContentProcessor
from src.llm.scenario_extractor import ScenarioExtractor
from src.llm.metric_calculator import MetricCalculator


def _extract_first_content_from_string(text: str) -> Optional[Dict]:
    from longterm.core.content_processor import extract_first_content_from_string
    result = extract_first_content_from_string(text)
    return result


def _parse_json_from_response(response: str) -> Optional[Dict]:
    """
    Unified JSON parsing function that handles both Markdown code blocks and direct JSON.
    
    Args:
        response: String response that may contain JSON (in Markdown or direct format)
        
    Returns:
        Parsed dictionary, or None if parsing fails
    """
    if not isinstance(response, str):
        # Already a dict
        return response if isinstance(response, dict) else None
    
    # Try to extract JSON from markdown code block first
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            print(f"[yellow]JSON parsing error from markdown: {e}[/yellow]")
    
    # Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"[yellow]Direct JSON parsing failed: {e}[/yellow]")
        return None

    
class longtermlossfunction:
    def __init__(self, scenario_description: str, llm_provider: str = "deepseek", temperature: float = 0, field_info: Optional[Dict] = None, risk_level: str = "high_risk", scene_graph_data=None):
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.scenario_description = scenario_description
        self.field_info = field_info
        self.risk_level = risk_level
        self.scene_graph_data = scene_graph_data # Store the numerical data
        self.analysis_agent = AnalysisAgent(provider=llm_provider, temperature=temperature)
        self.driver_agent = DriverAgent(provider=llm_provider, temperature=temperature)
        self.reflection_agent = ReflectionAgent(provider=llm_provider, temperature=temperature)
        
    def analysis_results(self):
        # Use the already created analysis_agent to avoid redundant instantiation
        behavior_analysis_raw, _ = self.analysis_agent.analyze_behavior(self.scenario_description, risk_level=self.risk_level, field_info=self.field_info)
        
        # Parse JSON using unified parsing function
        behavior_analysis = _parse_json_from_response(behavior_analysis_raw)
        if not behavior_analysis:
            print("[red]Failed to parse behavior analysis response[/red]")
            return None
        
        # Extract attacker_vehicle_id from selected_behavior in parsed JSON
        try:
            attacker_vehicle_id = behavior_analysis['selected_behavior']['attacker_vehicle_id']
            return int(attacker_vehicle_id.split('_')[-1])
        except (KeyError, AttributeError, ValueError) as e:
            print(f"[red]Failed to extract attacker_vehicle_id: {e}[/red]")
            return None

    def run_full_analysis(self):
        import time as _time
        from utils.logging_utils import Logger

        final_decision = {}
        self.timing = {}

        # Step 1: AnalysisAgent — 场景行为分析（LLM 调用）
        _t0 = _time.time()
        behavior_analysis_raw, _ = self.analysis_agent.analyze_behavior(
            self.scenario_description, risk_level=self.risk_level, field_info=self.field_info
        )
        self.timing['analysis_agent'] = _time.time() - _t0
        self.last_behavior_analysis_raw = behavior_analysis_raw

        behavior_analysis = _parse_json_from_response(behavior_analysis_raw)
        if not behavior_analysis:
            Logger.log('│      [LLM] AnalysisAgent JSON解析失败，返回空结果')
            return {}

        _analysis_api = getattr(self.analysis_agent, 'last_call_duration', None)
        _analysis_ttft = getattr(self.analysis_agent, 'last_first_token_time', None)
        Logger.log(f'│      AnalysisAgent 耗时 {self.timing["analysis_agent"]:.1f}s'
                   f'{f"  (API={_analysis_api:.1f}s  TTFT={_analysis_ttft:.1f}s)" if _analysis_api else ""}')

        # Step 1.5: LongTailPotentialAssessor — 确定性长尾评估（无LLM）
        longtail_assessment: Dict = {}
        behavior_label_for_lta = behavior_analysis.get("selected_behavior", {}).get("behavior_label", "")
        lta_input = self.scene_graph_data if self.scene_graph_data is not None else self.scenario_description
        try:
            lta = LongTailPotentialAssessor(min_potential=0.5)
            longtail_assessment = lta.assess(lta_input, behavior_label=behavior_label_for_lta)
            subtype = longtail_assessment.get("longtail_subtype", "")
            score   = longtail_assessment.get("potential_score", 0.0)
            if subtype:
                Logger.log(f'│      长尾评估: {subtype} (score={score:.2f})')
        except Exception as exc:
            longtail_assessment = {}

        # Step 2: MetricCalculator — 数值指标计算（主路径，无 LLM）
        _t0 = _time.time()
        selected_behavior = behavior_analysis.get("selected_behavior", {})
        agent_instructions = behavior_analysis.get("agent_instructions", {})

        raw_timing = selected_behavior.get("execution_timing", selected_behavior.get("timing_window", None))
        raw_params = selected_behavior.get("parameter_changes", selected_behavior.get("parameter_deltas", {}))
        if not raw_timing or not raw_params:
            cot = behavior_analysis.get("cot_reasoning", {})
            s4  = cot.get("step4_behavior_selection", {})
            raw_timing = raw_timing or s4.get("timing_window", s4.get("execution_timing", [0.5, 3.0]))
            raw_params = raw_params or s4.get("parameter_changes", s4.get("parameter_deltas", {}))

        timing_window = _parse_timing_window(raw_timing)
        param_changes = {
            "speed_delta_per_step":   _parse_numeric(raw_params.get("speed_delta_per_step",   0.0)),
            "heading_delta_per_step": _parse_numeric(raw_params.get("heading_delta_per_step", 0.0)),
        }
        target_vid = selected_behavior.get("target_vehicle_id", "ego_vehicle")
        if target_vid in ("vehicle_0", "0"):
            target_vid = "ego_vehicle"

        calculator_input = {
            "driver_agent_inputs": agent_instructions.get("driver_agent_inputs", {}),
            "key_interaction": {
                "attacker_vehicle_id": selected_behavior.get("attacker_vehicle_id", ""),
                "target_vehicle_id":   target_vid,
            },
            "adversarial_strategy": {"timing_window": timing_window, "parameter_changes": param_changes},
        }

        risk_metrics = {}
        try:
            calculator = MetricCalculator(self.scenario_description)
            risk_metrics = calculator.calculate_metrics(calculator_input)
        except Exception:
            risk_metrics = {}
        self.timing['metric_calculator'] = _time.time() - _t0

        # Step 2 回退: DriverAgent（LLM）
        self.timing['driver_agent'] = 0.0
        if not risk_metrics:
            Logger.log('│      MetricCalculator无结果，回退DriverAgent...')
            _t0 = _time.time()
            formatted_behavior_content = ContentProcessor.format_for_next_agent(
                behavior_analysis_raw, "AnalysisAgent")
            risk_metrics_raw = self.driver_agent.analyze_scenario_and_calculate_metrics(
                scenario_description=self.scenario_description,
                behavior_analysis={"full_content": formatted_behavior_content},
                risk_level=self.risk_level,
            )
            self.timing['driver_agent'] = _time.time() - _t0
            risk_metrics = _extract_first_content_from_string(risk_metrics_raw) if risk_metrics_raw else {}
            _driver_api = getattr(self.driver_agent, 'last_call_duration', None)
            _driver_ttft = getattr(self.driver_agent, 'last_first_token_time', None)
            Logger.log(f'│      DriverAgent 耗时 {self.timing["driver_agent"]:.1f}s'
                       f'{f"  (API={_driver_api:.1f}s  TTFT={_driver_ttft:.1f}s)" if _driver_api else ""}')

        # Step 3: ReflectionAgent — LLM生成损失权重
        _t0 = _time.time()
        final_decision_raw = self.reflection_agent.reflect_and_generate_weights(
            behavior_analysis={"full_content": behavior_analysis_raw},
            risk_metrics=risk_metrics,
            risk_level=self.risk_level,
            longtail_assessment=longtail_assessment,
        )
        self.timing['reflection_agent'] = _time.time() - _t0
        _reflect_api = getattr(self.reflection_agent, 'last_call_duration', None)
        _reflect_ttft = getattr(self.reflection_agent, 'last_first_token_time', None)
        Logger.log(f'│      ReflectionAgent 耗时 {self.timing["reflection_agent"]:.1f}s'
                   f'{f"  (API={_reflect_api:.1f}s  TTFT={_reflect_ttft:.1f}s)" if _reflect_api else ""}')

        if isinstance(final_decision_raw, dict) and "error" in final_decision_raw:
            Logger.log(f'│      ReflectionAgent返回错误: {final_decision_raw}')
            return {}

        final_decision = _parse_json_from_response(final_decision_raw)
        if not final_decision:
            Logger.log('│      ReflectionAgent JSON解析失败')
            return {}

        final_weights = final_decision.get("risk_weights", {})

        # 从 AnalysisAgent 输出中提取 attacker_vehicle_id
        attacker_vehicle_id = None
        try:
            if self.last_behavior_analysis_raw:
                from longterm.agents.analysis import LLM_analysis_results
                attacker_vehicle_id = LLM_analysis_results(self.last_behavior_analysis_raw)
        except Exception:
            pass

        atk_str = f'agt{attacker_vehicle_id}' if attacker_vehicle_id is not None else '未识别'
        _total_llm = self.timing['analysis_agent'] + self.timing.get('driver_agent', 0) + self.timing['reflection_agent']
        Logger.log(f'│      LLM分析完成  攻击者={atk_str}  权重={len(final_weights)}项'
                   f'  行为={selected_behavior.get("behavior_label", "?")[:30]}'
                   f'  [Analysis={self.timing["analysis_agent"]:.1f}s'
                   f'  Driver={self.timing.get("driver_agent", 0):.1f}s'
                   f'  Reflection={self.timing["reflection_agent"]:.1f}s'
                   f'  总LLM={_total_llm:.1f}s]')

        return {
            "risk_weights": final_weights,
            "attacker_vehicle_id": attacker_vehicle_id
        }


