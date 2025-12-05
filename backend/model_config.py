from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import httpx
import threading
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
import ollama

load_dotenv()

@dataclass
class ModelConfig:
    key: str
    model_name: str
    api_key: str
    endpoint: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved(self, value: str) -> Optional[str]:
        if isinstance(value, str) and value.startswith("env:"):
            env_var = value.split(":", 1)[1]
            return os.environ.get(env_var)
        return value

    @property
    def resolved_api_key(self) -> Optional[str]:
        return self.resolved(self.api_key)

    @property
    def resolved_endpoint(self) -> Optional[str]:
        return self.resolved(self.endpoint)

    @property
    def resolved_model(self) -> Optional[str]:
        return self.resolved(self.model_name)


class ModelRegistry:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ModelRegistry()
        return cls._instance

    def register(self, cfg: ModelConfig) -> None:
        self._models[cfg.key] = cfg

    def get(self, key: str) -> Optional[ModelConfig]:
        return self._models.get(key)


def load_defaults_into_registry(registry: Optional[ModelRegistry] = None) -> None:
    r = registry or ModelRegistry.get_instance()

    base_url = "env:TCS_GENAI_BASE_URL"
    api_key = "env:TCS_GENAI_API_KEY"

    r.register(
        ModelConfig(
            key="deepseek_v3",
            model_name="env:TCS_GENAI_MODEL_DEEPSEE3",
            api_key=api_key,
            endpoint=base_url,
            extra={"temperature": 0.3},
        )
    )

    # GPT-4o
    r.register(
        ModelConfig(
            key="gpt4o",
            model_name="env:TCS_GENAI_MODEL_GPT4O",
            api_key=api_key,
            endpoint=base_url,
            extra={"temperature": 0.2},
        )
    )

    # GPT-3.5 Turbo
    r.register(
        ModelConfig(
            key="gpt35",
            model_name="env:TCS_GENAI_MODEL_GPT35",
            api_key=api_key,
            endpoint=base_url,
            extra={"temperature": 0.3},
        )
    )

    # Phi-4 reasoning
    r.register(
        ModelConfig(
            key="phi_4",
            model_name="env:TCS_GENAI_MODEL_PHI_4",
            api_key=api_key,
            endpoint=base_url,
            extra={"temperature": 0.1},
        )
    )

    # Llama 3.3 70B Instruct
    r.register(
        ModelConfig(
            key="llama_70b",
            model_name="env:TCS_GENAI_MODEL_LLAMA_70B",
            api_key=api_key,
            endpoint=base_url,
            extra={"temperature": 0.2},
        )
    )


def get_client_for(key: str) -> ChatOpenAI:
    registry = ModelRegistry.get_instance()
    cfg = registry.get(key)

    if cfg is None:
        raise KeyError(f"Unknown model key: {key}")

    http_client = httpx.Client(verify=False)

    return ChatOpenAI(
        base_url=cfg.resolved_endpoint,
        model=cfg.resolved_model,
        api_key=cfg.resolved_api_key,
        http_client=http_client,
        **cfg.extra,
    )


_AUTO_LOAD = True
if _AUTO_LOAD:
    load_defaults_into_registry()


if __name__ == "__main__":
    client = get_client_for("deepseek_v3")
    resp = client.invoke("Hi")
    print(resp)