"""
加载和处理LLM权重配置。
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("ConfigLoader")

class ConfigLoader:
    """
    加载和处理LLM权重配置。
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration: {config_path}")
        return config

    """
    @staticmethod
    def get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:

        llm_config = config.get('llm', {})
        
        # 处理API密钥（优先使用环境变量）
        if 'api_key' not in llm_config:
            if llm_config.get('model_name', '').startswith('gpt') and "OPENAI_API_KEY" in os.environ:
                llm_config['api_key'] = os.environ["OPENAI_API_KEY"]
            elif (llm_config.get('model_name', '').startswith('claude') or 
                  llm_config.get('model_name', '').startswith('anthropic')) and "ANTHROPIC_API_KEY" in os.environ:
                llm_config['api_key'] = os.environ["ANTHROPIC_API_KEY"]
                
        return llm_config
    
    @staticmethod
    def get_static_weights(config: Dict[str, Any]) -> Dict[str, float]:
        
        weights = {}
        
        risk_functions = config.get('risk_functions', {})
        for key, risk_func in risk_functions.items():
            if risk_func.get('enabled', False):
                mapping = risk_func.get('mapping')
                default_weight = risk_func.get('default_weight', 0.0)
                
                if mapping:
                    weight_key = f"loss_{mapping}"
                    weights[weight_key] = default_weight
                    
        return weights
    
    @staticmethod
    def get_scenario_config(config: Dict[str, Any]) -> Dict[str, Any]:
      
        return config.get('scenario', {})
    
    @staticmethod
    def get_weight_update_config(config: Dict[str, Any]) -> Dict[str, Any]:
 
        return config.get('weight_update', {})
    
    @staticmethod
    def get_integration_config(config: Dict[str, Any]) -> Dict[str, Any]:

        return config.get('integration', {})
    """
    @staticmethod
    def is_llm_enabled(config: Dict[str, Any]) -> bool:
 
        return config.get('llm', {}).get('enabled', False)
    
    @staticmethod
    def setup_logging(config: Dict[str, Any]) -> None:
   
        log_level_str = config.get('integration', {}).get('log_level', 'INFO')
        log_level = getattr(logging, log_level_str, logging.INFO)
        

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"日志级别设置为: {log_level_str}")
    """
    @staticmethod
    def get_risk_function_descriptions(config: Dict[str, Any]) -> Dict[str, str]:
        
        descriptions = {}
        
        risk_functions = config.get('risk_functions', {})
        for key, risk_func in risk_functions.items():
            if 'description' in risk_func:
                descriptions[key] = risk_func['description']
                
        return descriptions 
    """