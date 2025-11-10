import os
import textwrap
import time
import json
from rich import print
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from longterm.core.llm_factory import BaseAgent
from src.llm.scenario_extractor import ScenarioExtractor


class AnalysisAgent(BaseAgent): 
    def __init__(
        self, temperature: float = 0, verbose: bool = False, provider: Optional[str] = None) -> None:
        super().__init__(temperature=temperature, verbose=verbose, provider=provider or "deepseek")
        self.behavior_corpus = self._load_behavior_corpus()
        self.behavior_escalation_strategies = self._load_behavior_escalation_strategies()
        self.risk_metrics_definitions = self._load_risk_metrics_definitions()
      
    def _load_behavior_corpus(self) -> Dict[str, Any]:
        corpus_path = os.path.join(
            os.path.dirname(__file__), '..', 'knowledge', 'behavior_corpus.json')
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
            print(f"Successfully loaded behavior corpus, containing {len(corpus['behavior_patterns'])} behavior patterns")
            return corpus
    
    def _load_behavior_escalation_strategies(self) -> str:
        strategies_path = os.path.join(
            os.path.dirname(__file__), '..', 'knowledge', 'behavior_escalation_strategies.md')
        with open(strategies_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Successfully loaded behavior escalation strategies document ({len(content)} characters)")
            return content
    
    def _load_risk_metrics_definitions(self) -> str:
        definitions_path = os.path.join(
            os.path.dirname(__file__), '..', 'knowledge', 'risk_metrics_definitions.md')
        with open(definitions_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Successfully loaded risk metrics definitions document ({len(content)} characters)")
            return content
    
    def _generate_few_shot_examples(self) -> str:    
        examples = []
        for behavior_name, behavior_data in self.behavior_corpus['behavior_patterns'].items():
            if behavior_data.get('few_shot_examples'):
                example = behavior_data['few_shot_examples'][0]
                examples.append(f"""
                【{behavior_name} - {behavior_data['description']}】
                - 典型场景: {example['nl_description']}
                - 识别要点: {', '.join(behavior_data['key_indicators'])}
                - 碰撞类型: {behavior_data['collision_type']}
                - 关键特征: {', '.join(example['structured_analysis']['key_features'])}
                """)
        
        return '\n'.join(examples[:3])
    
    def _get_behavior_matching_rules(self) -> str:
        rules = self.behavior_corpus['matching_rules']
        keyword_rules = []
        for keywords, behavior in rules.get('keyword_mapping', {}).items():
            keyword_rules.append(f"- Keyword \"{keywords}\" → {behavior}")
        
        return '\n'.join(keyword_rules)
    
    def _supports_vision(self) -> bool:
        vision_capable_models = [
            'gpt-4-vision-preview',
            'gpt-4-turbo',
            'gpt-4o',
            'gpt-4o-mini',
            'claude-3-opus',
            'claude-3-sonnet',
            'claude-3-haiku',
            'claude-3-5-sonnet',
            'gemini-pro-vision',
            'gemini-1.5-pro'
        ]
        
        model_name = getattr(self.llm, 'model_name', '').lower()
        
        for vision_model in vision_capable_models:
            if vision_model.lower() in model_name:
                return True
        
        return False
    
    
    def analyze_behavior(self, scenario_description: str, risk_level: Optional[str] = None, field_info: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Based on a complete scenario description, perform behavior analysis, integrating corpus knowledge and Few-shot examples.

        Parameters:
        - scenario_description: The complete scenario description obtained from extract_scenario_description, including:
        - Basic scenario information (number of vehicles, time steps, etc.)
        - Detailed vehicle information (position, speed, heading, dimensions, etc.)
        - Map and environmental information (road type, number of lanes, etc.)
        - Relative relationships between vehicles (direction, distance, etc.)
        - Potential risk identification (TTC, lateral distance, yaw rate, etc.)

        - Returns:
          - A tuple of (analysis result, original query)  
        """
        
        few_shot_examples = self._generate_few_shot_examples()
        matching_rules = self._get_behavior_matching_rules()

        system_message = textwrap.dedent("""\
        You are the **Chief Scenario Analysis Strategist** in the STRIVE system, responsible for executing the complete 5-step analysis process:
        Step 1: Scene Identification → Step 2: Metric Calculation → Step 3: Scene Generation → Step 4: Behavior Selection → Step 5: Reflection Summary

        **INPUT FORMAT SPECIFICATION**:
        You will receive structured JSON scenario data containing:
        - format_version, description, map name, dt (typically 0.5 seconds)
        - vehicles: List of vehicle objects with id, is_ego, type, length, width, trajectory, motion_analysis
        - Each trajectory contains timestamped waypoints with: t, x, y, heading, velocity, map_aligned
        - Time range: past (t ∈ [-2.0, 0.0]), future (t ∈ [0.5, 6.0]), dt = 0.5s per timestep
        - relative_motion_analysis: Pre-computed relative motion between ego and other vehicles (Chinese text)
        - dynamic_analysis: traffic_flow, complexity, risk_assessment (Chinese text)

        **CRITICAL DATA INTERPRETATION RULES**:
        1. **Vehicle Activity Assessment**:
           - Analyze velocity values across trajectory to determine if vehicle is active or stationary
           - Stationary: velocity ≈ 0 m/s consistently, or motion_analysis contains "近乎静止"
           - Active: velocity > 0.5 m/s with motion, suitable as potential attacker
           - Missing data: NaN values in trajectory indicate vehicle not present at that timestamp
        
        2. **Motion Analysis Field** (Chinese text format):
           - Acceleration: "加速(X m/s)" = accelerating, "减速" = decelerating, "匀速" = constant speed
           - Steering: "转向(X度)" = turning X degrees, "直行" = going straight
           - Speed state: "近乎静止" = nearly stationary (exclude from attackers)
        
        3. **Relative Motion Analysis** (Chinese text, CRITICAL for prioritization):
           - Format: "车辆X: 方位Ym, 运动趋势, 相对速度Zm/s"
           - "快速接近" = rapidly approaching → HIGH PRIORITY collision candidate
           - "快速远离" = rapidly departing → LOW PRIORITY
           - "保持距离" = maintaining distance → MEDIUM PRIORITY
           - Distance < 15m → Critical proximity, immediate attention required
           - Relative speed > 5 m/s → High collision potential
        
        4. **Physical Constraints** (dt = 0.5s):
           - Maximum acceleration: |a| ≤ 5 m/s² for typical vehicles
           - Velocity change per timestep: |Δv| ≤ 2.5 m/s (within 0.5s)
           - Heading change per timestep: |Δθ| ≤ 30° (realistic steering within 0.5s)
           - Maintain trajectory continuity (no sudden teleportation)

        **TEMPORAL ANALYSIS FRAMEWORK**:
        - Past trajectory (t ∈ [-2.0, 0.0]): Historical behavior reference, DO NOT modify
        - Current state (t = 0.5): Decision point and baseline state
        - Future prediction (t ∈ [0.5, 6.0]): PRIMARY target for adversarial modifications
        - Critical collision window: Typically t ∈ [1.0, 4.0]s for maximum impact

        **PRE-COMPUTED METRICS UTILIZATION**:
        - traffic_flow: Scene density and congestion level (e.g., "轻微拥堵, 平均速度2.4m/s")
        - complexity: Scene complexity level (e.g., "复杂场景, 高运动复杂度")
        - risk_assessment: Current baseline risk (e.g., "低风险场景") - your goal is to ESCALATE this
        - relative_motion_analysis: USE THIS to prioritize vehicles with "快速接近" status

        - **Step 1-2 Goal**: Deeply deconstruct the initial scene, identify key risks and safety indicators to monitor
        - **Step 3-4 Goal**: Identify the behavior strategy most likely to cause {risk_level} based on the {risk_level} risk level
        - **Step 5 Goal**: Verify the physical reasonableness and traffic common sense of the generated scene, provide a confidence score
        - **Overall Requirements**: Each step must have a complete COT reasoning process to ensure the logical chain is rigorous and traceable
        - **Output Purpose**: Provide precise, quantitative strategy guidance for DriverAgent's indicator calculation and ReflectionAgent's weight configuration

        [Standardized Behavior Tactical Handbook (Corpus Summary)]
        {few_shot_examples}

        [Tactical Identification Matching Rules]
        {matching_rules}

        {risk_metrics_definitions}

        [Behavior Escalation Strategy Reference]
        {behavior_escalation_strategies}

        [5-Step COT Strategy Reasoning Process]

        **Step 1: Scene Identification and Deconstruction (Scene Identification) - Corresponding to Process Diagram Step 1**
        ```
        Core Task: Identify all dynamic traffic participants based on vehicle trajectories and map information, and describe their basic states and road layouts
        Thinking Framework:
        - Scene Basic Elements Analysis: Vehicle number, time range [-2.0, 6.0]s, road geometric structure, traffic environment
        - Main Vehicle (Ego) Analysis: Current position, speed, heading, behavior mode, vulnerability assessment
        - Supporting Vehicle Classification:
          * Active vehicles: velocity > 0.5 m/s, no "近乎静止" in motion_analysis → Potential attackers
          * Stationary vehicles: velocity ≈ 0 or "近乎静止" → Static obstacles, exclude from attackers
          * Check for NaN trajectories: Vehicle not present, exclude from analysis
        - Collision Candidate Prioritization (USE relative_motion_analysis):
          * HIGH PRIORITY: Vehicles with "快速接近" status + distance < 15m
          * MEDIUM PRIORITY: Vehicles with "保持距离" or moderate relative speed
          * LOW PRIORITY: Vehicles with "快速远离" status
          * Parse format: "车辆X: 方位Ym, 运动趋势, 相对速度Zm/s"
        - Road Environment Modeling: Lane number, intersection type, traffic rules, environmental constraints
        - Initial Risk Point Identification: Based on relative_motion_analysis, predict conflict areas and time windows
        
        Output Requirements: [Complete list of scene participants with motion status, road layout description, prioritized risk assessment based on relative_motion_analysis]
        ```

        **Step 2: Key Indicator Calculation Guidance (Indicator Calculation) - Corresponding to Process Diagram Step 2**  
        ```
        Core Task: Provide precise safety indicator calculation priority and measurement point guidance to DriverAgent
        Thinking Framework:
        - Determine the key safety indicator priority based on the risk points in Step 1: TTC, MinDist_lat, YawRate, DeltaV, THW, etc.
        - Vehicle relationship mapping: Identify which vehicle pairs have the most critical indicator calculations
          * Prioritize ego vs. vehicles with "快速接近" status (from relative_motion_analysis)
          * Focus on active vehicles (velocity > 0.5 m/s)
          * Exclude stationary vehicles ("近乎静止") from TTC/THW calculations
        - Time window setting: 
          * Monitoring start: t = 0.5s (current state)
          * Critical window: t ∈ [1.0, 4.0]s (primary collision zone)
          * Calculation frequency: Every dt = 0.5s timestep
        - Threshold expectation setting: Based on experience and physical constraints, estimate the dangerous threshold range for each indicator
        - Resource allocation: Provide calculation priority ranking for DriverAgent based on relative_motion_analysis
        
        Output Requirements: [Provide a structured indicator calculation guidance list for DriverAgent, prioritized by relative_motion_analysis]
        ```

        **Step 3: Scene Generation Strategy (Scene Generation) - Corresponding to Process Diagram Step 3**
        ```
        Core Task: Generate the behavior strategy most likely to cause {risk_level} based on the {risk_level} risk level and vehicle relationships
        Thinking Framework:
        - Risk Level Matching: Convert {risk_level} into specific physical parameter adjustment targets
        - Tactical Handbook Matching: Match the opportunities identified in Step 1 with the standardized behavior tactical corpus
        - Attacker Vehicle Selection: 
          * ONLY select from active vehicles (velocity > 0.5 m/s, no "近乎静止")
          * Prioritize vehicles with "快速接近" status (from relative_motion_analysis)
          * Consider vehicles within critical distance < 15m for immediate threat
        - Core Conflict Identification: Identify the key factors preventing {risk_level} in the current prediction (such as safety distance, time buffer, etc.)
        - Optimization Path Design: 
          * Focus trajectory modifications on future timesteps t ∈ [0.5, 6.0]s
          * Target critical collision window t ∈ [1.0, 4.0]s for maximum impact
          * Respect physical constraints: |a| ≤ 5 m/s², |Δv| ≤ 2.5 m/s per 0.5s timestep, |Δθ| ≤ 30° per timestep
        - Physical Feasibility Verification: Ensure that the generated behavior is physically feasible under dt=0.5s timestep constraints
        
        Output Requirements: [Clear behavior generation strategy targeting active vehicles, including specific parameter adjustment plan for future trajectory t ∈ [0.5, 6.0]s]
        ```

        **Step 4: Optimal Behavior Selection (Behavior Selection) - Corresponding to Process Diagram Step 4**
        ```
        Core Task: From all possible long-tail scenarios, select the single best solution with the highest success probability
        Thinking Framework:
        - Candidate Solution Evaluation: Rank the multiple possible strategies generated in Step 3 based on success probability
        - Key Vehicle Determination: 
          * Select from active vehicles (velocity > 0.5 m/s, no "近乎静止") only
          * Prefer vehicles with "快速接近" status (from relative_motion_analysis)
          * Consider current distance and relative speed for collision timing
        - Behavior Label Assignment: Assign standardized behavior labels to the selected solution (such as 'AggressiveCutIn')
        - Execution Timing Optimization: 
          * Specify exact timesteps for trajectory modification (e.g., t ∈ [1.0, 3.5]s)
          * Ensure collision occurs within critical window t ∈ [1.0, 4.0]s
        - Post-Agent Instruction Generation: Provide specific execution guidance for DriverAgent and ReflectionAgent
        - **CRITICAL OUTPUT FORMAT**: attacker_vehicle_id MUST use format "vehicle_{{id}}" (e.g., "vehicle_2", NOT "Vehicle ID 2")
        
        Output Requirements: [Single best behavior solution with correct vehicle ID format, including clear execution instructions and parameter configuration]
        ```

        **Step 5: Reflection and Summary (Reflection and Summary) - Corresponding to Process Diagram Step 5**
        ```
        Core Task: Verify whether the generated scenario conforms to physical laws and traffic common sense, and provide the corresponding confidence score
        Thinking Framework:
        - Physical Law Check (dt = 0.5s): 
          * Acceleration constraints: |a| ≤ 5 m/s² for typical vehicles
          * Velocity changes: |Δv| ≤ 2.5 m/s per 0.5s timestep
          * Steering rate: |Δθ| ≤ 30° per 0.5s timestep (realistic steering)
          * Trajectory continuity: No sudden teleportation or discontinuities
        - Traffic Common Sense Validation: 
          * Check whether the generated behavior is within the reasonable driving behavior range
          * Verify that stationary vehicles ("近乎静止") remain stationary (don't suddenly start moving)
          * Ensure active vehicles maintain motion continuity from past trajectory
        - Logical Consistency Review: 
          * Ensure that the reasoning chain of Step 1-4 is logically consistent and without contradictions
          * Verify selected attacker is indeed an active vehicle (velocity > 0.5 m/s, no "近乎静止")
          * Confirm modifications target future timesteps t ∈ [0.5, 6.0]s only
        - Success Probability Evaluation: Based on all analysis, provide the expected success rate of {risk_level}
        - Risk Level Compliance Verification: Verify whether the final scenario truly matches the specified {risk_level}
        - Confidence Quantification: Provide a confidence score for the overall strategy within the range of 0-1
        
        Output Requirements: [Complete verification report with physical constraint checks, including physical reasonableness, traffic common sense, success probability and {risk_level} compliance score]

        ```json
        {{
          "cot_reasoning": {{
            "step1_scene_identification": {{
              "thinking": "I need to comprehensively identify all dynamic traffic participants in the scene based on vehicle trajectories and map information...",
              "logic": "By analyzing scene basic elements, ego vehicle state, other vehicles' behavior and road environment, I found...",  
              "conclusion": "Participant list: [specific list], Road layout: [specific description], Initial risk assessment: [specific analysis]"
            }},
            "step2_indicator_calculation": {{
              "thinking": "Based on Step 1 risk point identification, I need to provide precise metric calculation guidance for DriverAgent...",
              "logic": "By analyzing critical vehicle pair relationships and risk priorities, the most important safety metrics are...",
              "conclusion": "Key metric priorities: [specific ranking], Measurement points: [specific vehicle pairs], Threshold expectations: [specific value ranges]"
            }},
            "step3_scene_generation": {{
              "thinking": "Based on risk level and tactical matching, I need to formulate an adversarial scene generation strategy...",
              "logic": "By identifying core conflicts and optimization paths, the key physical parameter adjustment plan is...",
              "conclusion": "Adversarial strategy: [specific plan], Parameter adjustments: [specific values], Physical feasibility: [verification result]"
            }},
            "step4_behavior_selection": {{
              "thinking": "From all possible long-tail scenarios, I need to select the single solution with the highest success probability...",
              "logic": "By evaluating candidate solutions and determining the best attacker, the optimal choice is...",
              "conclusion": "Optimal behavior solution: [specific description], Execution vehicle: [vehicle ID], Behavior label: [standardized label]"
            }},
            "step5_reflection_summary": {{
              "thinking": "Finally, I need to verify the physical reasonableness and traffic common sense of the generated scenario, and provide a confidence score...",
              "logic": "Through physical law checks, traffic common sense validation, and logical consistency review, I found...",
              "conclusion": "Physical reasonableness: [check result], Traffic common sense: [validation result], Success probability: [percentage], Overall confidence: [0-1 score]"
            }}
          }},
          "scene_analysis_results": {{
            "identified_participants": ["ego_vehicle", "vehicle_1", "vehicle_2", "..."],
            "road_layout": "Detailed description of road type, number of lanes, intersection information, etc.",
            "initial_risk_assessment": "Risk assessment results based on current state"
          }},
          "indicator_guidance": {{
            "priority_metrics": ["TTC", "MinDist_lat", "YawRate", "DeltaV", "THW"],
            "critical_vehicle_pairs": ["ego_vehicle vs vehicle_X", "..."],
            "monitoring_timewindow": "Critical time window description",
            "threshold_expectations": {{
              "TTC": "< 2.0s",
              "MinDist_lat": "< 1.5m",
              "YawRate": "> 0.3 rad/s"
            }}
          }},
          "generated_strategy": {{
            "matched_behavior_label": "AggressiveCutIn|IntersectionRush|SuddenBraking|etc.",
            "matching_confidence": 0.98,
            "collision_type_primary": "rear_end|cut_in|intersection|lane_departure",
            "key_parameter_adjustments": {{
              "target_vehicle": "vehicle_ID",
              "speed_adjustment": "+/- X m/s",
              "lateral_adjustment": "+/- Y m",
              "timing_adjustment": "+/- Z s"
            }}
          }},
          "selected_behavior": {{
            "attacker_vehicle_id": "vehicle_2",  // MUST use format: vehicle_{{id}}, NOT "Vehicle ID X"
            "target_vehicle_id": "ego_vehicle",
            "interaction_type": "following|lane_change|intersection_crossing",
            "execution_timing": "Specific timestep range, e.g., t ∈ [1.0, 3.5]s",
            "success_probability": "high|medium|low"
          }},
          "reflection_validation": {{
            "physical_law_compliance": {{
              "acceleration_check": "compliant|non-compliant - specific explanation",
              "velocity_check": "compliant|non-compliant - specific explanation",
              "geometric_check": "compliant|non-compliant - specific explanation"
            }},
            "traffic_common_sense": {{
              "behavior_reasonableness": "reasonable|unreasonable - specific explanation",
              "scenario_realism": "realistic|unrealistic - specific explanation"
            }},
            "overall_confidence_score": 0.85,
            "risk_level_compliance": "Whether it matches the specified risk level",
            "success_probability_estimate": "Expected collision generation success rate percentage"
          }},
          "agent_instructions": {{
            "driver_agent_inputs": {{
              "priority_metrics": ["MinDist_lat", "YawRate", "TTC"],
              "measurement_points": "Calculate metrics for vehicle ID X relative to ego vehicle",
              "calculation_frequency": "Calculate every 0.1 seconds"
            }},
            "reflection_agent_inputs": {{
              "primary_risk_type": "lateral_cut_in|intersection_conflict|rear_end_collision",
              "behavior_label": "Matched standardized behavior label",
              "weight_adjustment_guidance": "Prioritize [list weight names in order] - NOTE: DO NOT include specific numerical values, only priority ordering",
              "loss_function_priorities": ["L_AdversarialCrash", "L_MinDist_lat", "L_YawRate"]
            }}
          }}
        }}
        ```
        """).format(
            risk_level=risk_level,
            few_shot_examples=few_shot_examples,
            matching_rules=matching_rules,
            risk_metrics_definitions=self.risk_metrics_definitions,
            behavior_escalation_strategies=self.behavior_escalation_strategies
        )

        # Build message list
        messages = [
            SystemMessage(content=system_message),
        ]
        
            
        human_message = f"""
            Please strictly follow your role as **Chief Strategist** and the **COT Strategy Reasoning Steps** to formulate a detailed adversarial generation strategy for the following traffic scenario.

            【Structured JSON Scenario Data】
            {scenario_description}

            【Critical Reminders】
            - Input format: This is structured JSON data, NOT natural language description
            - Parse motion_analysis field (Chinese text: "加速", "转向", "近乎静止", etc.)
            - **MUST utilize relative_motion_analysis** to identify collision candidates: "快速接近" = HIGH PRIORITY
            - Active vehicles: velocity > 0.5 m/s, no "近乎静止" → Potential attackers
            - Stationary vehicles: velocity ≈ 0 or "近乎静止" → Exclude from attackers
            - Target future trajectory modifications: t ∈ [0.5, 6.0]s, focus on t ∈ [1.0, 4.0]s
            - Physical constraints: dt = 0.5s, |a| ≤ 5 m/s², |Δv| ≤ 2.5 m/s per timestep, |Δθ| ≤ 30° per timestep

            【Your Tasks】
            1.  **Execute complete five-step strategy reasoning**: From scene identification to reflection summary, all steps are essential.
            2.  **Identify best executor and tactics**: 
                - Select ONLY from active vehicles (velocity > 0.5 m/s, no "近乎静止")
                - Prioritize vehicles with "快速接近" status (from relative_motion_analysis)
                - Specify which standard tactics should be adopted (e.g., 'AggressiveCutIn')
            3.  **Formulate specific optimization strategy**: 
                - Clearly explain how to change key physical quantities to cause {risk_level}
                - Respect physical constraints: dt = 0.5s, |a| ≤ 5 m/s², |Δv| ≤ 2.5 m/s, |Δθ| ≤ 30°
            4.  **Issue precise instructions**: Provide clear, executable follow-up guidance for `DriverAgent` and `ReflectionAgent`.
            5.  **Output standard JSON**: 
                - Ensure attacker_vehicle_id uses format "vehicle_{{id}}" (e.g., "vehicle_2")
                - Ensure the final output strictly conforms to the specified JSON format.
            """
            
        messages.append(HumanMessage(content=human_message))
        
        # Send request and get response
        print("---------------------------------------")
        print(scenario_description)
        print("---------------------------------------")
        print("[cyan]Performing knowledge-driven scenario behavior analysis...[/cyan]")
        response_content = ""
        try:
            for chunk in self.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if isinstance(content, str):
                    response_content += content
                    print(content, end="", flush=True)
                elif isinstance(content, list):
                    # Handle list type content
                    for item in content:
                        if isinstance(item, str):
                            response_content += item
                            print(item, end="", flush=True)
                        elif isinstance(item, dict):
                            # Handle dictionary type, extract text content
                            text_content = str(item.get('text', item.get('content', str(item))))
                            response_content += text_content
                            print(text_content, end="", flush=True)
        except Exception as e:
            print(f"[red]Streaming response processing error: {e}[/red]")
            # Try non-streaming call
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            response_content = str(content) if not isinstance(content, str) else content
        
        print("\n")
        
        # Get the last added message (should be HumanMessage)
        human_message = messages[-1] if len(messages) > 1 else None
        
        # Process human_message string representation
        if human_message is None:
            human_message_str = ""
        elif isinstance(human_message, HumanMessage):
            # HumanMessage object
            if isinstance(human_message.content, str):
                human_message_str = human_message.content
            elif isinstance(human_message.content, list):
                # Multimodal content, extract text parts
                text_parts = []
                for item in human_message.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                human_message_str = '\n'.join(text_parts)
            else:
                human_message_str = str(human_message.content)
        else:
            human_message_str = str(human_message)
        
        return str(response_content), human_message_str

# Get AnalysisAgent analysis results from response_content, extract attacker_vehicle_id from selected_behavior, and return the corresponding index
def LLM_analysis_results(response_content: str) -> int:
   """
   Extract attacker vehicle ID from AnalysisAgent's response.
   
   Args:
       response_content: Raw response string from AnalysisAgent (may contain Markdown or JSON)
       
   Returns:
       Integer vehicle ID extracted from the response
       
   Raises:
       ValueError: If parsing fails or required fields are missing
   """
   import re
   
   # Try to extract JSON from Markdown format first
   json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
   if json_match:
       json_str = json_match.group(1)
   else:
       # If ```json``` not found, try to parse directly
       json_str = response_content
   
   try:
       behavior_analysis = json.loads(json_str)
       selected_behavior = behavior_analysis['selected_behavior']
       attacker_vehicle_id = selected_behavior['attacker_vehicle_id']
       return int(attacker_vehicle_id.split('_')[-1])
   except (KeyError, AttributeError, ValueError, json.JSONDecodeError) as e:
       raise ValueError(f"Failed to extract attacker_vehicle_id: {e}")

