"""Application settings for MCP server."""

import os
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # Server settings
    server_name: str = "MCP Server"
    debug: bool = False
    
    # Tool settings
    tool_timeout: int = 30
    tool_max_retries: int = 3
    tool_retry_delay: float = 1.0
    
    # LLM settings
    openai_api_key: Optional[str] = None
    llm_endpoint: str = "https://api.openai.com/v1/chat/completions"
    llm_model: str = "gpt-3.5-turbo"
    llm_max_tokens: int = 100
    llm_temperature: float = 0.8
    
    # Social media settings
    opinion_update_interval: int = 30  # seconds
    max_opinions_per_company: int = 1000
    
    def __post_init__(self):
        """Load settings from environment variables."""
        self.debug = os.environ.get("DEBUG", "false").lower() == "true"
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.llm_endpoint = os.environ.get("LLM_ENDPOINT", self.llm_endpoint)
        self.llm_model = os.environ.get("LLM_MODEL", self.llm_model)
        
        # Parse integer settings
        if env_timeout := os.environ.get("TOOL_TIMEOUT"):
            self.tool_timeout = int(env_timeout)
        if env_retries := os.environ.get("TOOL_MAX_RETRIES"):
            self.tool_max_retries = int(env_retries)
        if env_interval := os.environ.get("OPINION_UPDATE_INTERVAL"):
            self.opinion_update_interval = int(env_interval)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings instance loaded from environment
    """
    return Settings()
