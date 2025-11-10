
import os
import json
import copy
import logging
from typing import Dict, Optional, Any, Union
from langchain_core.runnables.config import P
import torch
from utils.logging_utils import Logger

from src.llm.loss_function_agent import LossFunctionAgent
from src.llm.scenario_extractor import ScenarioExtractor 
import pdb

class WeightManager:
    
    def __init__(self, 
                 static_weights: Dict = None,
                 use_llm: bool = False,
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 temperature: float = 0.2,
                 cache_dir: Optional[str] = None,
                 traffic_model = None):
        self.static_weights = static_weights if static_weights is not None else {}
        self.base_weights = copy.deepcopy(self.static_weights)  # 保存原始权重作为基准，用于LLM缩放
        self.use_llm = use_llm
        self.cache_dir = cache_dir
        self.current_weights = copy.deepcopy(self.static_weights)
        self.traffic_model = traffic_model
        if use_llm:
            self.llm_agent = LossFunctionAgent(
                model_name=model_name,
                temperature=temperature,
                api_key=api_key,
                cache_dir=cache_dir
            )
                
        self.scenario_extractor = ScenarioExtractor(model=traffic_model)      
        self.weight_history = []
        self._record_weights("init")
        
    def get_weights(self) -> Dict:
        return copy.deepcopy(self.current_weights)
    
    def update_from_scenario(self, 
                            scene_graph, 
                            map_env=None, 
                            map_idx=None, 
                            future_pred=None,
                            past_traj=None, 
                            driving_objectives: Optional[str] = None,
                            extra_context: Optional[str] = None,
                            risk_level: str = "high_risk") -> Dict:
 
        Logger.log("=== Start dynamic scene generation weighting ===")

        scenario_json = self.scenario_extractor.extract_carla_scenario(
            scene_graph=scene_graph,
            map_env=map_env,
            map_idx=map_idx,
            past_traj=past_traj,
            future_pred=future_pred
        )
        scenario_description = json.dumps(scenario_json, ensure_ascii=False, indent=2)
        print(f"=== Scene description generation completed ===")
        self.last_scenario_description = scenario_description
        current_metrics = self._collect_current_metrics(scene_graph, future_pred)
        Logger.log(f"Collected {len(current_metrics)} metrics")
            
        # 调用LLM生成权重
        Logger.log("Calling longterm analysis process to generate weights...")
        llm_weights = self.llm_agent.generate_loss_weights(
                scenario_description=scenario_description,
                current_metrics=current_metrics,
                driving_objectives=driving_objectives,
                field_info=extra_context,
                risk_level=risk_level,
                scene_graph_data=scene_graph  # Pass the numerical data
            )
        Logger.log(f"Long-Term analysis completed, returned weights: {list(llm_weights.keys()) if llm_weights else 'None'}")
        
        # Debug: print full llm_weights structure
        Logger.log(f"[DEBUG] Full llm_weights content: {llm_weights}")
        
        # Save the LLM analysis results including attacker_vehicle_id
        self.last_llm_weights = llm_weights
        if llm_weights and 'attacker_vehicle_id' in llm_weights:
            self.attacker_vehicle_id = llm_weights['attacker_vehicle_id']
            Logger.log(f"✓ Saved attacker_vehicle_id: {self.attacker_vehicle_id}")
        else:
            Logger.log(f"✗ attacker_vehicle_id not found in llm_weights. Keys: {list(llm_weights.keys()) if llm_weights else 'None'}")
        
        # Try to save behavior analysis for debugging
        if hasattr(self.llm_agent, 'last_longterm_instance'):
            ltlf_instance = self.llm_agent.last_longterm_instance
            if hasattr(ltlf_instance, 'last_behavior_analysis_raw'):
                self.last_behavior_analysis_raw = ltlf_instance.last_behavior_analysis_raw
                Logger.log("✓ Saved behavior analysis raw data")
            
        if llm_weights and 'risk_weights' in llm_weights:
            self._update_weights_from_llm(llm_weights)
            self._record_weights(f"Update weights based on scene through longterm analysis")
                
            Logger.log("=== Weights updated successfully ===")

            risk_weights = llm_weights.get('risk_weights', {})
            for key, value in risk_weights.items():
                Logger.log(f"  {key}: {value}")
            
        return self.get_weights()
        
    def _collect_current_metrics(self, scene_graph, future_pred=None) -> Dict:
        
        metrics = {}
        if future_pred is not None:
            metrics["num_vehicles"] = scene_graph.past.size(0)
            metrics["num_scenes"] = scene_graph.batch.max().item() + 1
            
        return metrics
    
    def _update_weights_from_llm(self, llm_weights: Dict):
        if "risk_weights" in llm_weights:
            risk_weights = llm_weights["risk_weights"]
            mapping = {
                    "L_AdversarialCrash": "adv_crash",
                    "L_MinDist_lat": "min_dist_lat",
                    "L_TTC": "ttc",
                    "L_YawRate": "yaw_rate",
                    "L_VehicleCollision": "coll_veh",
                    "L_VehicleCollision_Planner": "coll_veh_plan",
                    "L_EnvironmentCollision": "coll_env",
                    "L_THW": "thw",
                    "L_DeltaV": "delta_v",
                    "L_TLC": "tlc",
                    "L_PathAdherence": "path_adherence",
                    "L_MotionBehavior": "motion_prior_atk",
                    "L_SceneSimilarity": "init_z_atk",
                    "L_Collision": "adv_crash",
                    "L_EnvironmentCollision_Planner": "coll_veh_plan",
                    "L_YawRate_Ego": "yaw_rate_ego",
                    "L_YawRate_NonEgo": "yaw_rate_non_ego",
                }
            for llm_key, weight_key in mapping.items():
                    if llm_key in risk_weights and weight_key in self.current_weights:
                        old_value = self.current_weights[weight_key]
                        llm_value = float(risk_weights[llm_key])
                        
                        # 检测并修正LLM生成的不合理权重值
                        # 核心攻击权重应该在10-100范围，如果LLM输出0-1范围，自动放大100倍
                        core_attack_keys = ['adv_crash', 'ttc', 'min_dist_lat', 'yaw_rate', 'delta_v']
                        supporting_keys = ['thw', 'tlc', 'path_adherence']
                        
                        if weight_key in core_attack_keys:
                            # 核心攻击权重最小应该是10.0
                            if llm_value < 2.0:  # 明显是0-1归一化值
                                Logger.log(f"⚠️  检测到归一化值 {llm_value} for {weight_key}，自动放大100倍")
                                new_value = llm_value * 100.0  # 0.95 → 95.0
                            elif llm_value < 10.0:  # 还是太小
                                Logger.log(f"⚠️  权重值 {llm_value} 小于最小值10.0，自动调整为10.0")
                                new_value = 10.0
                            else:
                                new_value = llm_value
                        elif weight_key in supporting_keys:
                            # 支持性权重最小应该是1.0
                            if llm_value < 1.0 and llm_value > 0.1:
                                Logger.log(f"⚠️  检测到归一化值 {llm_value} for {weight_key}，自动放大100倍")
                                new_value = llm_value * 100.0
                            elif llm_value < 1.0:
                                new_value = llm_value  # 0.1-1.0范围可能是合理的
                            else:
                                new_value = llm_value
                        else:
                            # 其他权重直接使用
                            new_value = llm_value
                        
                        self.current_weights[weight_key] = new_value
                        Logger.log(f"Update weights {weight_key}: {old_value} -> {new_value}")
   
    def _record_weights(self, description: str) -> None:
        self.weight_history.append({
            "description": description,
            "weights": copy.deepcopy(self.current_weights)
        })
