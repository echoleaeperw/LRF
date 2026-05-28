import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from rich import print
from langchain_core.messages import HumanMessage, SystemMessage

from longterm.core.llm_factory import BaseAgent
from longterm.prompts.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """
    Long-tail scenario analysis agent.

    Executes a 5-step COT reasoning process over the structured scenario JSON
    to identify adversarial opportunities and produce a behavior strategy.
    Prompts are loaded from longterm/prompts/analysis_system.md and analysis_human.md.
    """

    # Providers with large enough context to use full KB files (no compression needed)
    _LARGE_CONTEXT_PROVIDERS = {"gemini-flash", "gemini-pro"}

    def __init__(
        self,
        temperature: float = 0,
        verbose: bool = False,
        provider: Optional[str] = None,
    ) -> None:
        self._provider = provider or "deepseek"
        super().__init__(temperature=temperature, verbose=verbose, provider=self._provider)
        use_full_kb = self._provider in self._LARGE_CONTEXT_PROVIDERS
        self._knowledge = self._load_knowledge(full_kb=use_full_kb)

    # ------------------------------------------------------------------
    # Knowledge loading (corpus / strategies / definitions)
    # ------------------------------------------------------------------

    def _load_knowledge(self, full_kb: bool = False) -> Dict[str, str]:
        """
        Build KB injections for the system prompt.

        full_kb=True  → inject complete MD files (for large-context models like Gemini 1M)
        full_kb=False → inject compact tables only (~60 lines, for models with <128K context)
        """
        knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge")

        def _read(filename: str) -> str:
            path = os.path.join(knowledge_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        corpus_path = os.path.join(knowledge_dir, "behavior_corpus.json")
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        if full_kb:
            logger.info("Using FULL knowledge base (large-context provider)")
            return {
                "few_shot_examples":              self._build_behavior_catalog(corpus),
                "matching_rules":                 self._build_matching_rules(corpus),
                "risk_metrics_definitions":       _read("risk_metrics_definitions.md"),
                "behavior_escalation_strategies": _read("behavior_escalation_strategies.md"),
                "loss_functions_kb":              _read("loss_functions_kb.md"),
            }

        # Compact mode (default): tight tables ~60 lines total
        return {
            "few_shot_examples":            self._build_behavior_catalog(corpus),
            "matching_rules":               self._build_matching_rules(corpus),
            "risk_metrics_definitions":     self._build_risk_thresholds_compact(),
            "behavior_escalation_strategies": "",   # covered by loss_functions_kb compact
            "loss_functions_kb":            self._build_loss_priority_compact(corpus),
        }

    @staticmethod
    def _build_behavior_catalog(corpus: Dict) -> str:
        """
        Compact behavior catalog: one line per behavior showing
        collision type, primary metrics, and key trigger condition.
        """
        lines = ["## Behavior Catalog (all 9 types)\n",
                 "| Label | Collision | Primary Metrics | Key Trigger |",
                 "|-------|-----------|-----------------|-------------|"]
        for name, data in corpus.get("behavior_patterns", {}).items():
            metrics = data.get("priority_metrics", {})
            pri  = metrics.get("primary", "?")
            sec  = ", ".join(metrics.get("secondary", [])[:2])
            ctype = data.get("collision_type", "?")
            indics = data.get("key_indicators", [])
            trigger = indics[0] if indics else "?"
            lines.append(f"| {name} | {ctype} | {pri}, {sec} | {trigger} |")
        return "\n".join(lines)

    @staticmethod
    def _build_matching_rules(corpus: Dict) -> str:
        """Keyword → behavior type mapping (from corpus.matching_rules)."""
        rules = corpus.get("matching_rules", {})
        patterns = rules.get("keyword_patterns", {})   # fix: was 'keyword_mapping'
        if not patterns:
            return "(no keyword rules)"
        return "\n".join(f"- \"{kw}\" → {beh}" for kw, beh in patterns.items())

    @staticmethod
    def _build_risk_thresholds_compact() -> str:
        """Compact risk-level threshold table (replaces 188-line risk_metrics_definitions.md)."""
        return (
            "## Risk Thresholds\n"
            "| Metric        | high_risk      | longtail_condition |\n"
            "|---------------|----------------|--------------------|\n"
            "| TTC           | < 2.0s         | < 0.5s             |\n"
            "| MinDist_lat   | < 1.0m         | < 0.3m             |\n"
            "| YawRate       | > 15 deg/s     | > 25 deg/s         |\n"
            "| DeltaV        | > 15 m/s       | > 25 m/s           |\n"
            "| THW           | < 1.0s         | < 0.5s             |\n"
        )

    @staticmethod
    def _build_loss_priority_compact(corpus: Dict) -> str:
        """
        Compact loss-priority table per behavior type
        (replaces 517-line escalation strategies + 209-line loss KB).
        """
        mapping = corpus.get("loss_function_mapping", {})
        lines = [
            "## LLM-Controlled Losses (AdvGenLoss)\n",
            "Always set adv_crash HIGH. Choose ttc/min_dist_lat/yaw_rate by behavior type.\n",
            "yaw_rate has separate ego/non-ego controls: yaw_rate_ego (ego evasion) and yaw_rate_non_ego (attacker turns).\n",
            "Always RELAX motion_prior_atk and init_z_atk for the attacker.\n",
            "| Behavior | Priority Order | YawRate Ego/NonEgo | Relax |",
            "|----------|---------------|-------------------|-------|",
        ]
        relax_default = "motion_prior_atk, init_z_atk"
        templates = {
            "AggressiveCutIn":              ("adv_crash > min_dist_lat > yaw_rate > ttc",       "ego=LOW, non_ego=HIGH"),
            "SuddenBraking":                ("adv_crash > ttc > DeltaV",                        "ego=LOW, non_ego=LOW"),
            "AggressiveTailgating":         ("adv_crash > ttc > THW",                           "ego=LOW, non_ego=LOW"),
            "LaneDeparture":                ("adv_crash > min_dist_lat > yaw_rate",             "ego=HIGH, non_ego=HIGH"),
            "Overspeeding":                 ("adv_crash > DeltaV > ttc",                        "ego=LOW, non_ego=LOW"),
            "IntersectionRush":             ("adv_crash > ttc > min_dist_lat > yaw_rate",       "ego=MED, non_ego=HIGH"),
            "SuddenAcceleration":           ("adv_crash > DeltaV > ttc",                        "ego=LOW, non_ego=LOW"),
            "StationaryObstacleActivation": ("adv_crash > ttc > min_dist_lat",                  "ego=LOW, non_ego=MED"),
            "MultiVehiclePincer":           ("adv_crash > ttc > min_dist_lat > yaw_rate",       "ego=MED, non_ego=HIGH"),
        }
        for beh, (prio, yr_guide) in templates.items():
            relax = relax_default
            if beh == "LaneDeparture":
                relax += ", coll_env"
            if beh == "StationaryObstacleActivation":
                relax += " (extreme)"
            lines.append(f"| {beh} | {prio} | {yr_guide} | {relax} |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core analysis method
    # ------------------------------------------------------------------

    def analyze_behavior(
        self,
        scenario_json: str,
        risk_level: str = "high_risk",
        field_info: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """
        Run the 5-step COT analysis on a structured scenario JSON string.

        Args:
            scenario_json: JSON string from ScenarioExtractor.extract_structured_scenario()
            risk_level: Target risk level ("high_risk", "longtail_condition", "low_risk")
            field_info: Optional additional context (unused currently)

        Returns:
            (response_content: str, human_message_str: str)
        """
        system_content = PromptLoader.render(
            "analysis_system",
            risk_level=risk_level,
            **self._knowledge,
        )
        human_content = PromptLoader.render(
            "analysis_human",
            risk_level=risk_level,
            scenario_json=scenario_json,
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

        print(f"[cyan]AnalysisAgent: running 5-step COT (risk_level={risk_level})...[/cyan]")

        response_content = self._stream_llm(messages)

        return response_content, human_content

    # ------------------------------------------------------------------
    # LLM streaming helper
    # ------------------------------------------------------------------

    def _stream_llm(self, messages) -> str:
        """
        Stream or invoke the LLM, collecting the final response.

        Handles two streaming styles:
        - Delta streaming (OpenAI / DeepSeek): each chunk is a new token.
        - Cumulative streaming (some proxies like yibuapi): each chunk is the
          accumulated text so far. We detect this and print only the NEW portion.

        DeepSeek-R1: reasoning_content arrives in additional_kwargs (internal CoT).
        We print it lightly but only return content for downstream parsing.

        Sets self.last_call_duration (seconds) and self.last_first_token_time (seconds)
        for performance profiling.
        """
        import time as _time
        response = ""
        _printed_len = 0   # tracks how many chars we've already displayed
        _t_api_start = _time.time()
        _first_token_time = None

        def _print_new(text: str) -> None:
            """Print only the portion not yet displayed (handles cumulative chunks)."""
            nonlocal _printed_len
            if len(text) > _printed_len:
                new_part = text[_printed_len:]
                print(new_part, end="", flush=True)
                _printed_len = len(text)

        try:
            for chunk in self.llm.stream(messages):
                if _first_token_time is None:
                    _first_token_time = _time.time() - _t_api_start

                # --- DeepSeek-R1: internal reasoning chain ---
                if hasattr(chunk, "additional_kwargs"):
                    rc = chunk.additional_kwargs.get("reasoning_content", "")
                    if rc:
                        print(rc, end="", flush=True)   # reasoning is always delta

                piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                if isinstance(piece, str) and piece:
                    # Detect cumulative vs delta:
                    # If piece starts with the current response → cumulative proxy
                    if piece.startswith(response) and len(piece) > len(response):
                        response = piece          # replace, not append
                        _print_new(response)
                    else:
                        response += piece
                        _print_new(response)
                elif isinstance(piece, list):
                    for item in piece:
                        text = (
                            item if isinstance(item, str)
                            else str(item.get("text", item.get("content", str(item))))
                        )
                        response += text
                        _print_new(response)

        except Exception as e:
            logger.warning(f"Streaming failed, falling back to invoke: {e}")
            resp = self.llm.invoke(messages)
            response = resp.content if hasattr(resp, "content") else str(resp)
            if hasattr(resp, "additional_kwargs"):
                rc = resp.additional_kwargs.get("reasoning_content", "")
                if rc:
                    print(f"\n[dim][R1 内部推理][/dim] {rc[:300]}...")
            print(response)

        _api_elapsed = _time.time() - _t_api_start
        self.last_call_duration = _api_elapsed
        self.last_first_token_time = _first_token_time or _api_elapsed
        print(f"\n[dim]  ⏱ API call: {_api_elapsed:.1f}s (first token: {self.last_first_token_time:.1f}s)[/dim]")
        return response


# ------------------------------------------------------------------
# Standalone helper — extract attacker_vehicle_id from raw response
# ------------------------------------------------------------------

def LLM_analysis_results(response_content: str) -> int:
    """
    Extract the integer attacker vehicle index from AnalysisAgent's JSON response.

    Looks for selected_behavior.attacker_vehicle_id in the format "vehicle_X".

    Returns:
        Integer vehicle ID.

    Raises:
        ValueError: If parsing fails.
    """
    # Try to unwrap a ```json ... ``` block first
    json_match = re.search(r"```json\s*(.*?)\s*```", response_content, re.DOTALL)
    json_str = json_match.group(1) if json_match else response_content

    try:
        parsed = json.loads(json_str)
        attacker_id = parsed["selected_behavior"]["attacker_vehicle_id"]
        return int(attacker_id.split("_")[-1])
    except (KeyError, AttributeError, ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to extract attacker_vehicle_id: {e}")
