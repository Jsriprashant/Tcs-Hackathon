"""LLM and Embedding model configuration for TCS GenAI Lab."""

from functools import lru_cache
from typing import Optional
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .settings import get_settings


def get_http_client() -> httpx.Client:
    """
    Get configured HTTP client with SSL settings for corporate environments.
    
    Returns:
        Configured httpx.Client instance
    """
    settings = get_settings()
    
    # Configure SSL verification
    if settings.ssl_cert_path:
        # Use custom certificate
        verify = settings.ssl_cert_path
    elif settings.verify_ssl:
        # SSL verification enabled
        verify = True
    else:
        # Disable SSL verification (for development/testing only)
        verify = False
    
    return httpx.Client(verify=verify, timeout=60.0)


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Get configured LLM instance for TCS GenAI Lab.
    
    Args:
        model_name: Override default model
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens for response
        streaming: Enable streaming responses
    
    Returns:
        Configured ChatOpenAI instance
    """
    settings = get_settings()
    
    # Debug logging
    import logging
    logging.getLogger(__name__).info(f"Using model: {model_name or settings.default_model}")
    logging.getLogger(__name__).info(f"Base URL: {settings.tcs_genai_base_url}")
    logging.getLogger(__name__).info(f"SSL Verify: {settings.verify_ssl}")
    
    # Use custom HTTP client for SSL configuration
    http_client = get_http_client()
    
    return ChatOpenAI(
        model=model_name or settings.default_model,
        api_key=settings.tcs_genai_api_key,
        base_url=settings.tcs_genai_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        http_client=http_client,
    )


@lru_cache()
def get_embedding_model() -> Embeddings:
    """
    Get configured embedding model for TCS GenAI Lab.
    
    Returns:
        Configured OpenAIEmbeddings instance
    """
    settings = get_settings()
    
    # Use custom HTTP client for SSL configuration
    http_client = get_http_client()
    
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.tcs_genai_api_key,
        base_url=settings.tcs_genai_base_url,
        http_client=http_client,
    )


def get_reasoning_llm(temperature: float = 0.1) -> BaseChatModel:
    """
    Get LLM optimized for complex reasoning tasks (e.g., DeepSeek).
    
    Args:
        temperature: Sampling temperature
    
    Returns:
        Configured ChatOpenAI instance for reasoning
    """
    settings = get_settings()
    
    # Use custom HTTP client for SSL configuration
    http_client = get_http_client()
    
    return ChatOpenAI(
        model=settings.reasoning_model,
        api_key=settings.tcs_genai_api_key,
        base_url=settings.tcs_genai_base_url,
        temperature=temperature,
        max_tokens=8192,  # Larger context for reasoning
        http_client=http_client,
    )
