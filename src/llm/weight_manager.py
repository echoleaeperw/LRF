
import os
import json
import copy
import logging
from typing import Dict, Optional, Any, Union
import torch
from utils.logging_utils import Logger

from src.llm.loss_function_agent import LossFunctionAgent
from src.llm.scenario_extractor import ScenarioExtractor 
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
        # Ablation iii: accumulate per-scene LLM weight vectors for robustness reporting
        self.llm_weight_log: list = []
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
 
        scenario_json = self.scenario_extractor.extract_structured_scenario(
            scene_graph=scene_graph,
            map_env=map_env,
            map_idx=map_idx,
            past_traj=past_traj,
            future_pred=future_pred
        )
        scenario_description = json.dumps(scenario_json, ensure_ascii=False, indent=2)
        self.last_scenario_description = scenario_description
        current_metrics = self._collect_current_metrics(scene_graph, future_pred)

        llm_weights = self.llm_agent.generate_loss_weights(
                scenario_description=scenario_description,
                current_metrics=current_metrics,
                driving_objectives=driving_objectives,
                field_info=extra_context,
                risk_level=risk_level,
                scene_graph_data=scene_graph
            )

        # 保存攻击者ID
        self.last_llm_weights = llm_weights
        if llm_weights and 'attacker_vehicle_id' in llm_weights:
            self.attacker_vehicle_id = llm_weights['attacker_vehicle_id']
        else:
            self.attacker_vehicle_id = None

        # 保存行为分析原始数据
        if hasattr(self.llm_agent, 'last_longterm_instance'):
            ltlf_instance = self.llm_agent.last_longterm_instance
            if hasattr(ltlf_instance, 'last_behavior_analysis_raw'):
                self.last_behavior_analysis_raw = ltlf_instance.last_behavior_analysis_raw

        if llm_weights and 'risk_weights' in llm_weights:
            self._update_weights_from_llm(llm_weights)
            self._record_weights("longterm_analysis")

            # Ablation iii: 记录跨场景权重方差
            risk_weights = llm_weights.get('risk_weights', {})
            self.llm_weight_log.append({
                "scene_idx":           len(self.llm_weight_log),
                "risk_weights":        copy.deepcopy(risk_weights),
                "attacker_vehicle_id": llm_weights.get("attacker_vehicle_id", None),
            })

        return self.get_weights()
        
    def _collect_current_metrics(self, scene_graph, future_pred=None) -> Dict:
        
        metrics = {}
        if future_pred is not None:
            metrics["num_vehicles"] = scene_graph.past.size(0)
            metrics["num_scenes"] = scene_graph.batch.max().item() + 1
            
        return metrics
    
    def _update_weights_from_llm(self, llm_weights: Dict):
        """
        Direct replacement mode: ReflectionAgent now outputs weights using
        AdvGenLoss actual key names and actual value ranges. No scaling needed.
        """
        if "risk_weights" not in llm_weights:
            return

        risk_weights = llm_weights["risk_weights"]

        for key, value in risk_weights.items():
            if key in self.current_weights:
                old_value = self.current_weights[key]
                new_value = float(value)
                self.current_weights[key] = new_value
                Logger.log(f"Set {key}: {old_value} → {new_value}")
            else:
                Logger.log(f"Skip unknown weight key: {key}")
   
    def _record_weights(self, description: str) -> None:
        self.weight_history.append({
            "description": description,
            "weights": copy.deepcopy(self.current_weights)
        })

    def summarize_llm_weight_variance(self) -> Dict:
        """Ablation iii helper: compute cross-scene statistics of LLM-generated weights."""
        import numpy as np
        from collections import Counter

        if not self.llm_weight_log:
            return {"error": "No LLM weight log entries found"}

        all_rw = [entry["risk_weights"] for entry in self.llm_weight_log]
        attacker_ids = [entry["attacker_vehicle_id"] for entry in self.llm_weight_log]

        keys = sorted({k for rw in all_rw for k in rw})
        stats = {}
        for k in keys:
            vals = [float(rw[k]) for rw in all_rw if k in rw and rw[k] is not None]
            if not vals:
                continue
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            cv   = std / mean if mean != 0 else float("inf")
            stats[k] = {"mean": round(mean, 4), "std": round(std, 4), "cv": round(cv, 4)}

        id_counter = Counter(str(a) for a in attacker_ids)
        most_common_pair = id_counter.most_common(1)
        most_count = most_common_pair[0][1] if most_common_pair else 0
        attacker_stability = round(most_count / len(attacker_ids), 3) if attacker_ids else 0.0

        Logger.log("=== LLM Weight Variance Summary (Ablation iii) ===")
        for k, v in stats.items():
            Logger.log(f"  {k:<22}  mean={v['mean']:.4f}  std={v['std']:.4f}  CV={v['cv']:.4f}")
        Logger.log(f"  attacker_id_stability = {attacker_stability:.3f}")

        return {
            "n_scenes":           len(self.llm_weight_log),
            "per_weight_stats":   stats,
            "attacker_stability": attacker_stability,
            "attacker_id_distribution": dict(id_counter),
        }
