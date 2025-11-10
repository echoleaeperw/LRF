import os
import textwrap
import time
import json
import re
import math
from rich import print
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.scenario_extractor import ScenarioExtractor
from longterm.core.llm_factory import BaseAgent

class DriverAgent(BaseAgent):
    """
    Calculate key safety metrics based on AnalysisAgent's strategy analysis and scenario data.
    """
    def __init__(
        self, temperature: float = 0, verbose: bool = False, provider: Optional[str] = None
    ) -> None:
        super().__init__(temperature=temperature, verbose=verbose, provider=provider)

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Use unified robust JSON parser
        """
        from longterm.core.json_parser import RobustJSONParser
        result = RobustJSONParser.extract_json_from_response(text)
        if result:
            return RobustJSONParser.validate_and_normalize_weights(result)
        return None

    def analyze_scenario_and_calculate_metrics(
        self, scenario_description: str, behavior_analysis: Dict[str, Any], risk_level: str = "high_risk"
    ) -> Dict[str, Any]:
        """
        Calculate key safety metrics based on scenario description and behavior analysis.
        
        Args:
            scenario_description: Complete scenario JSON data (including vehicle trajectories)
            behavior_analysis: Macro behavior analysis results from AnalysisAgent
            risk_level: Target risk level ("low_risk", "high_risk", "longtail_condition")
            
        Returns:
            A dictionary containing calculated metrics (JSON format)
        """
        system_message = textwrap.dedent(f"""\
        You are the **Tactical Analyst** in the STRIVE system, responsible for executing the second step of the adversarial scenario generation pipeline: **Precise Quantitative Metric Calculation**.

        【Core Mission】
        - **Role**: Interface with AnalysisAgent's output, execute actual quantitative calculations and risk assessment
        - **Main Objective**: Transform AnalysisAgent's qualitative strategy analysis into precise quantitative metric data, providing decision basis for ReflectionAgent
        - **Target Risk Level**: Current task aims to generate scenarios conforming to **{risk_level}** risk level
        - **Core Principle**: **"Precise Calculation"**. Based on actual vehicle trajectory data, use standard traffic safety metric calculation formulas

        【Input Data Format Specification】
        You will receive two types of data:
        
        1. **AnalysisAgent's Strategy Analysis Results (JSON format)**:
           - `agent_instructions.driver_agent_inputs.priority_metrics`: List of priority metrics that must be calculated
           - `generated_strategy.collision_type_primary`: Core risk type
           - `indicator_guidance.critical_vehicle_pairs`: Critical vehicle pair relationships
           - `indicator_guidance.threshold_expectations`: Threshold expectations for each metric
           - `selected_behavior.attacker_vehicle_id`: Attacker vehicle ID (format: "vehicle_X")
           - `selected_behavior.target_vehicle_id`: Target vehicle ID (usually "ego_vehicle")
        
        2. **Structured Scenario JSON Data**:
           ```json
           {{
             "dt": 0.5,
             "vehicles": [
               {{
                 "id": 0,
                 "is_ego": true,
                 "type": "ego_vehicle",
                 "length": 4.084,
                 "width": 1.730,
                 "trajectory": [
                   {{"t": -2.0, "x": 1010.13, "y": 1414.19, "heading": -82.88, "velocity": 4.19}},
                   {{"t": -1.5, "x": 1010.26, "y": 1412.10, "heading": -91.98, "velocity": 4.19}},
                   ...
                 ],
                 "motion_analysis": "Accelerating(1.2m/s), Turning(56.2 degrees)"
               }},
               ...
             ],
             "relative_motion_analysis": "Vehicle 1: 13.4m ahead, rapidly approaching, relative speed -5.4m/s; ..."
           }}
           ```

        【Key Safety Metrics and Their Precise Calculation Formulas】
        Based on the actual implementation of STRIVE loss functions, here are the standard calculation methods for each metric:

        **1. TTC (Time-to-Collision)**
        - **Physical Meaning**: How long until collision occurs if current relative velocity remains constant
        - **Calculation Formula**: 
          ```
          Step 1: Calculate longitudinal relative distance d_rel_long
            - Extract ego and target positions at the same timestep (x, y)
            - Calculate position difference: Δx = target_x - ego_x, Δy = target_y - ego_y
            - Extract ego's heading vector (hx, hy) = (cos(heading), sin(heading))
            - Longitudinal distance: d_rel_long = Δx * hx + Δy * hy
          
          Step 2: Calculate longitudinal relative velocity v_rel_long
            - Calculate velocity from trajectory: v = (pos[t+1] - pos[t]) / dt
            - Extract ego and target velocity vectors (vx, vy)
            - Longitudinal relative velocity: v_rel_long = (target_vx - ego_vx) * hx + (target_vy - ego_vy) * hy
          
          Step 3: Calculate TTC
            - If v_rel_long <= 0 (not approaching): TTC = inf
            - Otherwise: TTC = d_rel_long / v_rel_long
          ```
        - **Risk Assessment**: TTC < 3.0s is high risk, TTC < 2.0s is extremely high risk

        **2. MinDist_lat (Minimum Lateral Distance)**
        - **Physical Meaning**: Lateral clearance between vehicle edges
        - **Calculation Formula**:
          ```
          Step 1: Calculate lateral distance d_lat
            - Extract ego and target positions (x, y)
            - Calculate position difference: Δx = target_x - ego_x, Δy = target_y - ego_y
            - Extract ego's heading vector (hx, hy)
            - Calculate perpendicular vector: (perp_x, perp_y) = (-hy, hx)
            - Lateral distance: d_lat = |Δx * perp_x + Δy * perp_y|
          
          Step 2: Calculate clearance d_lat_gap
            - Extract vehicle widths: ego_width, target_width
            - Clearance: d_lat_gap = d_lat - (ego_width + target_width) / 2
          ```
        - **Risk Assessment**: d_lat_gap < 0.5m is high risk, d_lat_gap < 0 indicates overlap (collision)

        **3. YawRate (Yaw Rate) - Steering Aggressiveness**
        - **Physical Meaning**: Rate of vehicle heading change, reflecting steering aggressiveness
        - **Calculation Formula**:
          ```
          Step 1: Extract heading angle sequence from trajectory
            - heading[t] in degrees
          
          Step 2: Calculate angle change rate
            - Δheading = heading[t+1] - heading[t]
            - Handle angle wraparound: if |Δheading| > 180°, then Δheading -= sign(Δheading) * 360°
            - yaw_rate = Δheading / dt (unit: degrees/second)
          ```
        - **Risk Assessment**: |yaw_rate| > 15°/s is aggressive steering, |yaw_rate| > 30°/s is extreme steering

        **4. THW (Time Headway)**
        - **Calculation Formula**: THW = d_rel_long / ego_velocity
        - **Risk Assessment**: THW < 1.5s is high risk

        **5. RelativeSpeed**
        - **Calculation Formula**: v_rel = sqrt((target_vx - ego_vx)^2 + (target_vy - ego_vy)^2)
        - **Risk Assessment**: |v_rel| > 5 m/s is high relative speed

        【Execution Tasks】

        **Task 1: Extract Key Information from AnalysisAgent Output**
        - Identify the priority metrics list in `priority_metrics`
        - Extract `attacker_vehicle_id` and `target_vehicle_id`
        - Extract `critical_vehicle_pairs` and `threshold_expectations`
        - Understand target risk level: {risk_level}

        **Task 2: Extract Trajectory Information from Scenario JSON Data**
        - Parse `vehicles` array to find attacker and target vehicles
        - Extract their `trajectory` data (t, x, y, heading, velocity)
        - Extract vehicle dimensions (length, width)
        - Extract time step `dt`

        **Task 3: Execute Precise Metric Calculations**
        - For each metric in `priority_metrics`, execute calculations according to the above formulas
        - Select appropriate time window (typically future trajectory t ∈ [0.5, 4.0]s)
        - Calculate metric values at each time step
        - Find minimum/maximum values (depending on metric type)
        - Record complete calculation process

        **Task 4: Risk Assessment and Summary**
        - Based on calculated values and threshold expectations, determine risk level for each metric
        - Evaluate whether current scenario meets {risk_level} target
        - Provide optimization recommendations for ReflectionAgent

        【Output Format Requirements】
        Please strictly follow the following JSON format for output:

        ```json
        {{
          "extracted_info": {{
            "target_risk_level": "{risk_level}",
            "collision_type": "Extracted from collision_type_primary",
            "priority_metrics": ["TTC", "MinDist_lat", "YawRate", ...],
            "attacker_vehicle_id": "vehicle_X",
            "target_vehicle_id": "ego_vehicle",
            "critical_vehicle_pairs": ["ego_vehicle vs vehicle_X"],
            "threshold_expectations": {{
              "TTC": "< 2.0s",
              "MinDist_lat": "< 1.5m",
              "YawRate": "> 15 deg/s"
            }},
            "behavior_label": "Extracted from matched_behavior_label"
          }},
          "trajectory_data": {{
            "dt": 0.5,
            "time_window": "t ∈ [0.5, 4.0]s",
            "attacker_vehicle": {{
              "id": "vehicle_X",
              "length": 4.25,
              "width": 1.64,
              "sample_trajectory_points": [
                {{"t": 0.5, "x": 1031.60, "y": 1422.10, "heading": 15.42, "velocity": 7.51}},
                {{"t": 1.0, "x": 1035.06, "y": 1423.55, "heading": 22.63, "velocity": 7.51}}
              ]
            }},
            "target_vehicle": {{
              "id": "ego_vehicle",
              "length": 4.08,
              "width": 1.73,
              "sample_trajectory_points": [
                {{"t": 0.5, "x": 1008.68, "y": 1406.31, "heading": -111.11, "velocity": 4.25}},
                {{"t": 1.0, "x": 1007.74, "y": 1404.40, "heading": -116.10, "velocity": 4.25}}
              ]
            }}
          }},
          "calculated_metrics": {{
            "TTC": {{
              "values_over_time": [
                {{"t": 0.5, "d_rel_long": 25.3, "v_rel_long": 3.26, "ttc": 7.76}},
                {{"t": 1.0, "d_rel_long": 23.7, "v_rel_long": 3.26, "ttc": 7.27}},
                {{"t": 1.5, "d_rel_long": 22.1, "v_rel_long": 3.26, "ttc": 6.78}}
              ],
              "min_value": 6.78,
              "min_value_time": 1.5,
              "threshold": "< 3.0s",
              "risk_level": "medium",
              "risk_explanation": "Minimum TTC is 6.78s, above safety threshold 3.0s, but decreasing over time, potential risk exists"
            }},
            "MinDist_lat": {{
              "values_over_time": [
                {{"t": 0.5, "d_lat": 18.5, "d_lat_gap": 15.5}},
                {{"t": 1.0, "d_lat": 17.2, "d_lat_gap": 14.2}}
              ],
              "min_value": 14.2,
              "min_value_time": 1.0,
              "threshold": "< 1.5m",
              "risk_level": "low",
              "risk_explanation": "Minimum lateral clearance is 14.2m, far exceeding danger threshold 1.5m, laterally safe"
            }},
            "YawRate": {{
              "values_over_time": [
                {{"t": 0.5, "heading_change": 7.21, "yaw_rate": 14.42}},
                {{"t": 1.0, "heading_change": 5.47, "yaw_rate": 10.94}}
              ],
              "max_value": 14.42,
              "max_value_time": 0.5,
              "threshold": "> 15 deg/s",
              "risk_level": "medium",
              "risk_explanation": "Maximum yaw rate is 14.42°/s, approaching aggressive steering threshold 15°/s"
            }}
          }},
          "risk_assessment": {{
            "high_risk_metrics": ["List high-risk metrics and reasons"],
            "medium_risk_metrics": ["List medium-risk metrics and reasons"],
            "low_risk_metrics": ["List low-risk metrics and reasons"],
            "overall_risk_level": "medium",
            "risk_level_compliance": "Current scenario's overall risk level is medium, analysis of compliance with target {risk_level}",
            "most_critical_metric": "TTC",
            "most_critical_value": 6.78,
            "key_contradiction": "Explain main obstacles or opportunities for achieving {risk_level} target"
          }},
          "recommendations_for_reflection_agent": {{
            "focus_metrics": ["TTC", "YawRate"],
            "optimization_direction": "Need to increase TTC weight to encourage closer following, while increasing YawRate weight to generate more aggressive steering",
            "suggested_weight_adjustments": {{
              "increase": ["ttc", "yaw_rate"],
              "decrease": ["min_dist_lat"],
              "reasoning": "Specific adjustment recommendations based on current metric values"
            }}
          }}
        }}
        ```

        【Critical Reminders】
        1. **Must extract actual trajectory data from scenario JSON** for calculations, do not fabricate values
        2. **Strictly follow formulas** for calculations, show complete calculation process
        3. **Handle edge cases**: NaN values, division by zero, angle wraparound, etc.
        4. **Time window**: Focus on future trajectory t ∈ [0.5, 4.0]s
        5. **Output JSON format**: Ensure it can be parsed by Python's json.loads()
        6. **Do not specify weight values**, only provide optimization recommendations, specific weights determined by ReflectionAgent
        """)
        
        # Process behavior_analysis parameter, extract actual content
        if isinstance(behavior_analysis, dict) and "full_content" in behavior_analysis:
            behavior_content = behavior_analysis["full_content"]
        else:
            behavior_content = str(behavior_analysis)
        
        human_message = textwrap.dedent(f"""\
        Please execute precise quantitative metric calculations based on AnalysisAgent's strategy analysis results and structured scenario data.

        【Target Risk Level】
        Current task aims to generate scenarios conforming to **{risk_level}** risk level.

        【AnalysisAgent Complete Analysis Results (JSON format)】
        {behavior_content}

        【Structured Scenario JSON Data (including complete vehicle trajectories)】
        {scenario_description}

        【Your Tasks】
        1. **Extract Key Information**: Extract priority metrics, vehicle IDs, thresholds, etc. from AnalysisAgent's JSON output
        2. **Parse Trajectory Data**: Extract complete trajectories of attacker and target vehicles from scenario JSON
        3. **Execute Precise Calculations**: For each priority metric, calculate according to standard formulas, show complete process
        4. **Risk Assessment**: Evaluate overall risk level based on calculation results
        5. **Provide Recommendations**: Provide optimization direction recommendations for ReflectionAgent
        6. **Output JSON**: Ensure final output is in standard JSON format

        【Special Reminders】
        - Extract actual x, y, heading, velocity data from scenario JSON's `vehicles[i].trajectory`
        - Use `dt` value from scenario JSON for time-related calculations
        - For metrics that cannot be calculated (e.g., missing data), mark as null in JSON and explain reason
        - All numerical calculations must be based on actual data, do not fabricate
        - Remember: goal is to achieve {risk_level} risk level
        """)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        print("[cyan]Calculating risk metrics...[/cyan]")
        response_content = ""
        try:
            for chunk in self.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if isinstance(content, str):
                    response_content += content
                    print(content, end="", flush=True)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            response_content += item
                            print(item, end="", flush=True)
                        elif isinstance(item, dict):
                            text_content = str(item.get('text', item.get('content', str(item))))
                            response_content += text_content
                            print(text_content, end="", flush=True)
        except Exception as e:
            print(f"[red]Streaming response processing error: {e}[/red]")
            try:
                response = self.llm.invoke(messages)
                response_content = response.content if hasattr(response, 'content') else str(response)
                if not isinstance(response_content, str):
                    response_content = str(response_content)
            except Exception as invoke_e:
                print(f"[red]LLM invocation failed: {invoke_e}[/red]")
                return {"error": "Failed to get response from LLM", "details": str(invoke_e)}

        print("\n[green]Risk metric calculation completed.[/green]")
        
        # Try to parse JSON, return raw text if parsing fails
        try:
            parsed_result = self._extract_json_from_response(response_content)
            if parsed_result:
                return parsed_result
            else:
                print("[yellow]Warning: Unable to parse JSON, returning raw text[/yellow]")
                return {"raw_response": response_content}
        except Exception as e:
            print(f"[yellow]JSON parsing failed: {e}, returning raw text[/yellow]")
            return {"raw_response": response_content}

