import os
import json
import re
from rich import print
from typing import Dict, Optional
from longterm.agents.analysis import AnalysisAgent
from longterm.agents.driver import DriverAgent
from longterm.agents.reflection import ReflectionAgent
from longterm.core.content_processor import ContentProcessor
from src.llm.scenario_extractor import ScenarioExtractor
from src.llm.metric_calculator import MetricCalculator # Import the new calculator


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
        print("[bold magenta]=== Starting Long-Term Analysis Process ===[/bold magenta]")
        print(f"[yellow]Using LLM Provider: {self.llm_provider}, Temperature: {self.temperature}[/yellow]")
        
        final_decision = {}

        # 1. Use already created Agents (initialized in __init__)
        print("\n[bold cyan]Step 1/3: Scene Behavior Analysis[/bold cyan]")
        print("[yellow]AnalysisAgent is analyzing scenario behavior patterns...[/yellow]")
        behavior_analysis_raw, _ = self.analysis_agent.analyze_behavior(self.scenario_description, risk_level=self.risk_level, field_info=self.field_info)
        
        # Save original analysis results for later use (avoid redundant LLM calls)
        self.last_behavior_analysis_raw = behavior_analysis_raw
        
        # Extract JSON from raw output using unified parsing function
        behavior_analysis = _parse_json_from_response(behavior_analysis_raw)
        if not behavior_analysis:
            print("[red]Failed to parse behavior analysis response[/red]")
            return {}
        print("[green]✓ Successfully parsed JSON from behavior analysis[/green]")
        
        # Debug: Check if agent_instructions exists
        if "agent_instructions" in behavior_analysis:
            print(f"[green]✓ Found agent_instructions with keys: {list(behavior_analysis['agent_instructions'].keys())}[/green]")
        else:
            print("[red]✗ agent_instructions NOT found in behavior_analysis[/red]")
            print(f"[yellow]Available keys: {list(behavior_analysis.keys())}[/yellow]")
        
        print("\n[green]✓ Scene Behavior Analysis Complete[/green]")

        # 3. Step 2: Risk Metric Calculation (New precise method or fallback to old method)
        print("\n" + "="*60)
        print("[bold cyan]Step 2/3: Risk Metric Calculation[/bold cyan]")
        print("="*60)
        
        risk_metrics_raw = None
        risk_metrics = {}

        if self.scene_graph_data is not None:
            print("[yellow]Using precise MetricCalculator with numerical data...[/yellow]")
            try:
                calculator = MetricCalculator(self.scene_graph_data)
                
                # Transform behavior_analysis to match MetricCalculator's expected format
                # MetricCalculator expects: driver_agent_inputs, key_interaction
                # AnalysisAgent provides: agent_instructions.driver_agent_inputs, selected_behavior
                calculator_input = {
                    "driver_agent_inputs": behavior_analysis.get("agent_instructions", {}).get("driver_agent_inputs", {}),
                    "key_interaction": {
                        "attacker_vehicle_id": behavior_analysis.get("selected_behavior", {}).get("attacker_vehicle_id", ""),
                        "target_vehicle_id": behavior_analysis.get("selected_behavior", {}).get("target_vehicle_id", "ego_vehicle")
                    }
                }
                
                risk_metrics = calculator.calculate_metrics(calculator_input)
                # For compatibility, we can format a raw string if needed by ReflectionAgent
                risk_metrics_raw = f"```json\n{json.dumps(risk_metrics, indent=2)}\n```"
                print("\n[green]✓ Precise Risk Metric Calculation Complete[/green]")
            except Exception as e:
                print(f"[red]MetricCalculator failed: {e}. Falling back to DriverAgent.[/red]")
                import traceback
                print(f"[yellow]Traceback:\n{traceback.format_exc()}[/yellow]")
                risk_metrics = {} # Reset on failure
        
        # Fallback to DriverAgent if numerical data is not present or calculator fails
        if not risk_metrics:
            print("[yellow]DriverAgent is calculating safety risk metrics...[/yellow]")
            
            # Pass full behavior analysis content to DriverAgent
            formatted_behavior_content = ContentProcessor.format_for_next_agent(
                behavior_analysis_raw, "AnalysisAgent")
            risk_metrics_raw = self.driver_agent.analyze_scenario_and_calculate_metrics(
                scenario_description=self.scenario_description,
                behavior_analysis={"full_content": formatted_behavior_content},
                risk_level=self.risk_level
            )
            # Apply content processing approach for DriverAgent output as well
            risk_metrics = _extract_first_content_from_string(risk_metrics_raw) if risk_metrics_raw else {}
            
            
            
        # 4. Step 3: Reflection and Weight Generation (based on Step 2 results)
        print("\n" + "="*60)
        print("[bold cyan]Step 3/3: Reflection and Weight Generation[/bold cyan]")
        print("="*60)
        print("[yellow]ReflectionAgent is synthesizing analysis and generating weights...[/yellow]")
        
        # Prepare complete input content for ReflectionAgent
        all_analysis_content = behavior_analysis_raw
        if risk_metrics_raw:
            # Ensure risk_metrics_raw is a string before processing
            if isinstance(risk_metrics_raw, dict):
                # Convert dict to JSON string
                risk_metrics_raw = f"```json\n{json.dumps(risk_metrics_raw, indent=2)}\n```"
            elif not isinstance(risk_metrics_raw, str):
                # Convert other types to string
                risk_metrics_raw = str(risk_metrics_raw)
            
            # Use the raw metrics string (either from calculator or DriverAgent)
            all_analysis_content += "\n\n" + ContentProcessor.format_for_next_agent(
                risk_metrics_raw, "MetricCalculator" if self.scene_graph_data else "DriverAgent"
            )
        
        final_decision_raw = self.reflection_agent.reflect_and_generate_weights(
            behavior_analysis={"full_content": all_analysis_content},
            risk_metrics=risk_metrics,  # Pass the structured metrics dict
            risk_level=self.risk_level
        )
        
        # Check if ReflectionAgent returned an error
        if isinstance(final_decision_raw, dict) and "error" in final_decision_raw:
            print(f"[red]ReflectionAgent返回错误: {final_decision_raw}[/red]")
            return {}
        
        # ReflectionAgent now returns JSON directly, parse if needed using unified function
        final_decision = _parse_json_from_response(final_decision_raw)
        if not final_decision:
            print(f"[red]JSON解析失败，无法从ReflectionAgent获取有效数据[/red]")
            print(f"[yellow]返回内容: {str(final_decision_raw)[:500]}...[/yellow]")
            return {}
        print("[green]✓ Successfully parsed JSON from ReflectionAgent[/green]")
            
        # 5. Extract and return final weights
        final_weights = final_decision.get("risk_weights", {})
        
        # Extract attacker_vehicle_id from behavior analysis
        attacker_vehicle_id = None
        print("\n[cyan]=== Extracting attacker_vehicle_id from behavior analysis ===[/cyan]")
        import sys
        sys.stdout.flush()  # Force flush to ensure output is captured
        try:
            if self.last_behavior_analysis_raw:
                print(f"[yellow]last_behavior_analysis_raw exists, length: {len(self.last_behavior_analysis_raw)}[/yellow]")
                sys.stdout.flush()
                from longterm.agents.analysis import LLM_analysis_results
                attacker_vehicle_id = LLM_analysis_results(self.last_behavior_analysis_raw)
                print(f"[green]✓ Extracted attacker_vehicle_id: {attacker_vehicle_id}[/green]")
                sys.stdout.flush()
            else:
                print("[red]✗ last_behavior_analysis_raw is empty or None[/red]")
                sys.stdout.flush()
        except Exception as e:
            print(f"[yellow]⚠ Failed to extract attacker_vehicle_id: {e}[/yellow]")
            import traceback
            print(f"[red]Traceback:\n{traceback.format_exc()}[/red]")
            sys.stdout.flush()
                                                                                                                                                                                                                  
        if final_weights:
            print("\n[green]✓ Weight Generation Complete[/green]")
            print(f"[blue]Generated {len(final_weights)} loss function weights[/blue]")
        else:
            print("\n[yellow]⚠ Weight generation may have issues[/yellow]")
            
        print("\n" + "="*60)
        print("[bold magenta]=== Long-Term Analysis Process Complete ===[/bold magenta]")
        print("="*60)
        
        # Return both weights and attacker_vehicle_id
        return {
            "risk_weights": final_weights,
            "attacker_vehicle_id": attacker_vehicle_id
        }


