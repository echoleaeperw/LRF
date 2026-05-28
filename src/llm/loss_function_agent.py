import json
import os
import re
import time
from typing import Dict, List, Optional, Union, Any
import logging
from longterm.agents.flow import longtermlossfunction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LossFunctionAgent")

class LossFunctionAgent:
    def __init__(self, model_name: str = "deepseek", temperature: float = 0.2, api_key: Optional[str] = None, verbose: bool = False, cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            
    
    def generate_loss_weights(self, 
                             scenario_description: str,
                             llm_provider: Optional[str] = None,
                             temperature: Optional[float] = None,
                             current_metrics: Optional[Dict] = None,
                             driving_objectives: Optional[str] = None,
                             field_info: Optional[Dict] = None,
                             risk_level: str = "high_risk",
                             scene_graph_data: Optional[Any] = None,
                            ) -> Dict:
        """
        Get the loss function from the last result of longterm
        
        Args:
            scenario_description: The scenario description
            llm_provider: LLM provider (if None, use self.model_name)
            temperature: Temperature parameter (if None, use self.temperature)
            current_metrics: Current metrics (not used)
            driving_objectives: Driving objectives (not used)
            field_info: Field information, including images and statistics (directly passed to longterm analysis)
            risk_level: Target risk level ("low_risk", "high_risk", "longtail_condition")
            scene_graph_data: The numerical scene graph data object.
            
        Returns:
            A dictionary of risk weights, format: {"risk_weights": {...}}
        """
        provider = llm_provider if llm_provider is not None else self.model_name
        temp = temperature if temperature is not None else self.temperature

        if provider.startswith("gpt"):
            provider = "openai"
        elif provider.startswith("claude") or provider.startswith("anthropic"):
            provider = "claude"
        elif provider.startswith("deepseek"):
            provider = "deepseek"

        # 缓存读取（cache_dir=None 时禁用）
        cache_key = cache_path = None
        if self.cache_dir is not None:
            cache_key = self._generate_cache_key(scenario_description)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    return json.load(f)

        ltlf = longtermlossfunction(
            scenario_description=scenario_description,
            llm_provider=provider,
            temperature=temp,
            field_info=field_info,
            risk_level=risk_level,
            scene_graph_data=scene_graph_data
        )
        loss_weights = ltlf.run_full_analysis()
        self.last_longterm_instance = ltlf

        result = self._process_longterm_results(loss_weights)

        # 写入缓存（cache_dir=None 时跳过）
        if self.cache_dir is not None and cache_key is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=2)

        return result

    
    def _process_longterm_results(self, loss_weights) -> Dict:
        """
        Process the results of longterm analysis, ensuring the format is correct
        
        Args:
            loss_weights: The results of longterm analysis
            
        Returns:
            A dictionary of formatted weights
        """
        logger.info("Processing longterm analysis results...")
        
        if loss_weights is None:
            logger.warning("Long-Term analysis returned None")
            return {"risk_weights": {}}
        
        if isinstance(loss_weights, dict):
            if "risk_weights" in loss_weights:
                logger.info("Detected standard risk_weights format")
                return loss_weights

            advgen_keys = {"adv_crash", "ttc", "min_dist_lat", "yaw_rate",
                           "yaw_rate_ego", "yaw_rate_non_ego",
                           "coll_veh", "coll_veh_plan", "coll_env",
                           "motion_prior_atk", "init_z_atk"}

            if any(k in loss_weights for k in advgen_keys):
                logger.info("Detected AdvGenLoss key format, wrapping as risk_weights")
                return {"risk_weights": loss_weights}

            logger.info("Wrapping unknown format as risk_weights")
            return {"risk_weights": loss_weights}
        
        else:
            logger.warning(f"Long-Term analysis returned non-dictionary type: {type(loss_weights)}")
            logger.warning(f"Returned content: {loss_weights}")
            return {"risk_weights": {}}
    
    def _generate_cache_key(self, scenario_description: str) -> str:
        """Generate a unique key for caching"""
        import hashlib
        return hashlib.md5(scenario_description.encode()).hexdigest()
    
    