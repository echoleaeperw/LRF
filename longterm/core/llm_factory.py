import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic.v1 import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic


class FullResponseCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.full_response = ""
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.full_response += token

class BaseAgent(ABC):
    
    def __init__(self, temperature: float = 0, verbose: bool = False, provider: Optional[str] = "deepseek"):
        self.verbose = verbose
        self.config = self._load_config()
        self.llm = self._initialize_llm(provider, temperature)

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'llm_config.json')
        with open(config_path, 'r') as f:
            return json.load(f)

    def _initialize_llm(self, provider: Optional[str], temperature: float):
        effective_provider = provider or self.config.get("default_provider", "deepseek")
        
        callbacks = []
        if self.verbose:
            callbacks.append(StreamingStdOutCallbackHandler())
        
        provider_config = self.config.get("providers", {}).get(effective_provider, {})
        api_key = provider_config.get("api_key") or os.getenv(f"{effective_provider.upper()}_API_KEY")
        base_url = provider_config.get("base_url")

        if not api_key:
            raise ValueError(f"API Key not found for {effective_provider}. Please check llm_config.json or set environment variables.")

        if effective_provider == "openai":
            model_name = provider_config.get("model", "gpt-4-turbo")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url=base_url,
                callbacks=callbacks,
            )
        elif effective_provider == "deepseek":
            model_name = provider_config.get("model", "deepseek-chat")
            deepseek_base_url = provider_config.get("base_url", "https://api.deepseek.com/v1")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url=deepseek_base_url,
                callbacks=callbacks,
            )
        elif effective_provider in ["anthropic", "claude"]:
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Claude/Anthropic support requires installing langchain_anthropic: pip install langchain-anthropic")
            
            model_name = provider_config.get("model", "claude-3-sonnet-20240229")
            anthropic_base_url = provider_config.get("base_url", "https://api.anthropic.com")
            
            return ChatAnthropic(
                model=model_name,
                
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url=anthropic_base_url,
                callbacks=callbacks,
            )
        elif effective_provider == "azure":
            pass
        else:
            raise ValueError(f"Unsupported LLM provider: {effective_provider}")

    def get_full_response_callback(self):
        return FullResponseCallbackHandler() 