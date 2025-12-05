# filepath: c:\Users\GenAIBLRANCUSR23\Desktop\Hackathon\backend\src\config\settings.py
"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Optional, List
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses pydantic-settings to automatically load and validate
    environment variables from .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # =========================================================================
    # TCS GenAI Lab Configuration
    # =========================================================================
    tcs_genai_base_url: str = Field(
        default="https://genailab.tcs.in",
        alias="TCS_GENAI_BASE_URL",
        description="TCS GenAI Lab API base URL"
    )
    
    tcs_genai_api_key: str = Field(
        default="",
        alias="TCS_GENAI_API_KEY",
        description="TCS GenAI Lab API key"
    )
    
    # Default model (GPT-4o)
    default_model: str = Field(
        default="azure/genailab-maas-gpt-4o",
        alias="TCS_GENAI_MODEL_GPT4O",
        description="Default LLM model"
    )
    
    # Reasoning model (DeepSeek)
    reasoning_model: str = Field(
        default="azure_ai/genailab-maas-DeepSeek-R1",
        alias="TCS_GENAI_MODEL_DEEPSEEK_R1",
        description="Reasoning model for complex tasks"
    )
    
    # Embedding model
    embedding_model: str = Field(
        default="azure/genailab-maas-text-embedding-3-large",
        alias="TCS_GENAI_MODEL_EMBED",
        description="Text embedding model"
    )
    
    # Additional models (optional)
    model_gpt35: str = Field(
        default="azure/genailab-maas-gpt-35-turbo",
        alias="TCS_GENAI_MODEL_GPT35",
    )
    model_gpt4o_mini: str = Field(
        default="azure/genailab-maas-gpt-4o-mini",
        alias="TCS_GENAI_MODEL_GPT4O_MINI",
    )
    model_whisper: str = Field(
        default="azure/genailab-maas-whisper",
        alias="TCS_GENAI_MODEL_WHISPER",
    )
    
    # =========================================================================
    # SSL Configuration
    # =========================================================================
    verify_ssl: bool = Field(
        default=False,
        alias="VERIFY_SSL",
        description="Enable/disable SSL verification"
    )
    
    ssl_cert_path: Optional[str] = Field(
        default=None,
        alias="SSL_CERT_PATH",
        description="Path to custom SSL certificate"
    )
    
    # =========================================================================
    # ChromaDB Configuration
    # =========================================================================
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        alias="CHROMA_PERSIST_DIR",
        description="ChromaDB persistence directory"
    )
    
    chroma_collection_prefix: str = Field(
        default="dd_",
        alias="CHROMA_COLLECTION_PREFIX",
        description="Prefix for ChromaDB collections"
    )
    
    # =========================================================================
    # MCP Server Configuration
    # =========================================================================
    MCP_SERVER_URL: Optional[str] = Field(
        default=None,
        description="MCP Server URL for tool connectivity"
    )
    
    mcp_api_key: Optional[str] = Field(
        default=None,
        alias="MCP_API_KEY",
        description="MCP Server API key"
    )
    
    mcp_timeout: int = Field(
        default=30,
        alias="MCP_TIMEOUT",
        description="MCP request timeout in seconds"
    )
    
    mcp_max_retries: int = Field(
        default=3,
        alias="MCP_MAX_RETRIES",
        description="Maximum MCP request retries"
    )
    
    # =========================================================================
    # Risk Scoring Configuration
    # =========================================================================
    risk_weight_financial: float = Field(
        default=0.35,
        alias="RISK_WEIGHT_FINANCIAL",
    )
    risk_weight_legal: float = Field(
        default=0.30,
        alias="RISK_WEIGHT_LEGAL",
    )
    risk_weight_hr: float = Field(
        default=0.15,
        alias="RISK_WEIGHT_HR",
    )
    risk_weight_strategic: float = Field(
        default=0.20,
        alias="RISK_WEIGHT_STRATEGIC",
    )
    
    # =========================================================================
    # Logging Configuration
    # =========================================================================
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    log_format: str = Field(
        default="json",
        alias="LOG_FORMAT",
        description="Log format (json, text)"
    )
    
    langsmith_tracing: bool = Field(
        default=False,
        alias="LANGSMITH_TRACING",
    )
    
    # =========================================================================
    # Security & Guardrails
    # =========================================================================
    enable_pii_filter: bool = Field(
        default=True,
        alias="ENABLE_PII_FILTER",
    )
    
    enable_content_moderation: bool = Field(
        default=True,
        alias="ENABLE_CONTENT_MODERATION",
    )
    
    max_input_length: int = Field(
        default=10000,
        alias="MAX_INPUT_LENGTH",
    )
    
    max_output_length: int = Field(
        default=50000,
        alias="MAX_OUTPUT_LENGTH",
    )
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    app_name: str = Field(
        default="MA Due Diligence Orchestrator",
        alias="APP_NAME",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0",
        alias="APP_VERSION",
        description="Application version"
    )
    
    debug: bool = Field(
        default=False,
        alias="DEBUG",
        description="Enable debug mode"
    )
    
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment (development, staging, production)"
    )
    
    # =========================================================================
    # Server Configuration
    # =========================================================================
    host: str = Field(
        default="0.0.0.0",
        alias="HOST",
        description="Server host"
    )
    
    port: int = Field(
        default=8123,
        alias="PORT",
        description="Server port"
    )
    
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="ALLOWED_ORIGINS",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # =========================================================================
    # Agent Configuration
    # =========================================================================
    max_tool_calls: int = Field(
        default=15,
        alias="MAX_TOOL_CALLS",
        description="Maximum tool calls per agent run"
    )
    
    agent_timeout: int = Field(
        default=500,
        alias="AGENT_TIMEOUT",
        description="Agent timeout in seconds"
    )
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Get allowed origins as a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached singleton).
    
    Uses lru_cache to ensure the same Settings instance is returned
    on subsequent calls, avoiding repeated environment parsing.
    
    Returns:
        Settings: Configured application settings
    """
    return Settings()
