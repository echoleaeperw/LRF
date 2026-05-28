import asyncio
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic.v1 import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import fastapi_poe as fp
    POE_AVAILABLE = True
except ImportError:
    POE_AVAILABLE = False


class _ChatPoe:
    """
    Minimal LangChain-compatible wrapper for POE API (fastapi_poe).

    POE's get_bot_response is an AsyncGenerator; this class bridges it to the
    synchronous invoke() / stream() interface that BaseAgent._stream_llm expects.

    Proxy support: set proxy_url in llm_config.json providers.poe.proxy_url,
    or set the HTTP_PROXY / HTTPS_PROXY environment variable.
    """

    def __init__(self, api_key: str, bot_name: str,
                 temperature: Optional[float] = None,
                 proxy_url: Optional[str] = None):
        self.api_key = api_key
        self.bot_name = bot_name
        self.temperature = temperature
        # proxy: explicit > env vars
        self.proxy_url = proxy_url or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_poe(messages) -> list:
        """Convert LangChain message objects to POE ProtocolMessages."""
        result = []
        for msg in messages:
            cls = type(msg).__name__
            if cls == "SystemMessage":
                role = "system"
            elif cls in ("HumanMessage", "human"):
                role = "user"
            elif cls in ("AIMessage", "ai"):
                role = "bot"
            else:
                role = "user"
            result.append(fp.ProtocolMessage(role=role, content=msg.content))
        return result

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    def _make_session(self):
        """Build an httpx.AsyncClient with optional proxy support."""
        try:
            import httpx
            if self.proxy_url:
                return httpx.AsyncClient(proxies={"https://": self.proxy_url,
                                                   "http://":  self.proxy_url})
            return httpx.AsyncClient()
        except ImportError:
            return None

    async def _acollect(self, messages) -> str:
        full = ""
        poe_msgs = self._to_poe(messages)
        session = self._make_session()
        kwargs = dict(
            messages=poe_msgs,
            bot_name=self.bot_name,
            api_key=self.api_key,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if session is not None:
            kwargs["session"] = session
        try:
            async for partial in fp.get_bot_response(**kwargs):
                if hasattr(partial, "text"):
                    full += partial.text
        finally:
            if session is not None:
                await session.aclose()
        return full

    # ------------------------------------------------------------------
    # Sync interface (mirrors LangChain ChatModel)
    # ------------------------------------------------------------------

    def invoke(self, messages):
        content = asyncio.run(self._acollect(messages))

        class _Resp:
            def __init__(self, c):
                self.content = c

        return _Resp(content)

    def stream(self, messages):
        """Yield chunks one at a time (sync adapter over async generator)."""
        poe_msgs = self._to_poe(messages)

        async def _gen():
            session = self._make_session()
            kwargs = dict(messages=poe_msgs, bot_name=self.bot_name, api_key=self.api_key)
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if session is not None:
                kwargs["session"] = session
            try:
                async for partial in fp.get_bot_response(**kwargs):
                    yield partial
            finally:
                if session is not None:
                    await session.aclose()

        loop = asyncio.new_event_loop()
        agen = _gen()

        class _Chunk:
            def __init__(self, c):
                self.content = c

        try:
            while True:
                partial = loop.run_until_complete(agen.__anext__())
                yield _Chunk(getattr(partial, "text", ""))
        except StopAsyncIteration:
            pass
        finally:
            loop.close()


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
        # 允许在 llm_config.json 里单独配置 max_tokens，默认 8192
        max_tokens = provider_config.get("max_tokens", 8192)

        if not api_key:
            raise ValueError(f"API Key not found for {effective_provider}. Please check llm_config.json or set environment variables.")

        if effective_provider == "siliconflow":
            model_name = provider_config.get("model", "Pro/zai-org/GLM-4.7")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url=provider_config.get("base_url", "https://api.siliconflow.cn/v1"),
                max_tokens=max_tokens,
                callbacks=callbacks,
            )
        elif effective_provider in ["openai", "gemini-flash", "gemini-pro"]:
            model_name = provider_config.get("model", "gpt-4-turbo")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url=base_url,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )
        elif effective_provider in ["deepseek", "deepseek-r1"]:
            model_name = provider_config.get("model", "deepseek-chat")
            deepseek_base_url = provider_config.get("base_url", "https://api.deepseek.com/v1")
            extra = {}
            # deepseek-reasoner 不支持 temperature 参数
            if model_name == "deepseek-reasoner":
                extra["temperature"] = 1   # R1 只接受 temperature=1
            else:
                extra["temperature"] = temperature
            return ChatOpenAI(
                model=model_name,
                api_key=SecretStr(api_key),
                base_url=deepseek_base_url,
                max_tokens=max_tokens,
                callbacks=callbacks,
                **extra,
            )
        elif effective_provider in ["anthropic", "claude"]:
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Claude/Anthropic support requires installing langchain_anthropic: pip install langchain-anthropic")
            
            model_name = provider_config.get("model", "claude-3-sonnet-20240229")
            anthropic_base_url = provider_config.get("base_url", "https://api.anthropic.com")
            # ChatAnthropic 内部会追加 /v1，所以去掉末尾多余的 /v1
            if anthropic_base_url.endswith("/v1"):
                anthropic_base_url = anthropic_base_url[:-3]

            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url=anthropic_base_url,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )
        elif effective_provider == "poe":
            if not POE_AVAILABLE:
                raise ValueError("POE support requires: pip install fastapi-poe")
            bot_name = provider_config.get("bot_name", "Claude-3.7-Sonnet")
            proxy_url = provider_config.get("proxy_url")
            return _ChatPoe(
                api_key=api_key,
                bot_name=bot_name,
                temperature=temperature if temperature != 0 else None,
                proxy_url=proxy_url,
            )
        elif effective_provider == "azure":
            pass
        else:
            raise ValueError(f"Unsupported LLM provider: {effective_provider}")

    def get_full_response_callback(self):
        return FullResponseCallbackHandler() 