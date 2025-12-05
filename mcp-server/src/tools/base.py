"""Base utilities for tool implementation."""

import time
from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.common.logging_config import get_logger, log_tool_call
from src.common.errors import ToolError, ExternalServiceError
from src.config.settings import get_settings

logger = get_logger(__name__)

T = TypeVar("T")


def tool_wrapper(
    name: str | None = None,
    log_params: bool = True,
    log_result: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to wrap tool functions with logging and error handling.
    
    Args:
        name: Optional tool name override
        log_params: Whether to log input parameters
        log_result: Whether to log the result
    
    Returns:
        Decorated function
    
    Example:
        @tool_wrapper(name="my_tool")
        def my_tool_impl(param1: str) -> str:
            return f"Result: {param1}"
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        tool_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            params = kwargs if log_params else {"[redacted]": True}
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                log_tool_call(
                    logger,
                    tool_name,
                    params,
                    result=result if log_result else "[redacted]",
                    duration_ms=duration_ms,
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_tool_call(
                    logger,
                    tool_name,
                    params,
                    error=e,
                    duration_ms=duration_ms,
                )
                raise
        
        return wrapper
    return decorator


def async_tool_wrapper(
    name: str | None = None,
    log_params: bool = True,
    log_result: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async decorator to wrap tool functions with logging and error handling.
    
    Args:
        name: Optional tool name override
        log_params: Whether to log input parameters
        log_result: Whether to log the result
    
    Returns:
        Decorated async function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        tool_name = name or func.__name__
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            params = kwargs if log_params else {"[redacted]": True}
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                log_tool_call(
                    logger,
                    tool_name,
                    params,
                    result=result if log_result else "[redacted]",
                    duration_ms=duration_ms,
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_tool_call(
                    logger,
                    tool_name,
                    params,
                    error=e,
                    duration_ms=duration_ms,
                )
                raise
        
        return wrapper
    return decorator


def with_retry(
    max_attempts: int | None = None,
    retry_exceptions: tuple = (ExternalServiceError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add retry logic to a function.
    
    Args:
        max_attempts: Maximum number of attempts (defaults to settings)
        retry_exceptions: Tuple of exceptions to retry on
    
    Returns:
        Decorated function with retry logic
    """
    settings = get_settings()
    attempts = max_attempts or settings.tool_max_retries
    
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(retry_exceptions),
        reraise=True,
    )


def validate_required_params(
    params: dict[str, Any],
    required: list[str],
    tool_name: str = "unknown",
) -> None:
    """
    Validate that required parameters are present.
    
    Args:
        params: Dictionary of parameters
        required: List of required parameter names
        tool_name: Name of the tool for error messages
    
    Raises:
        ToolError: If any required parameters are missing
    """
    missing = [p for p in required if p not in params or params[p] is None]
    if missing:
        raise ToolError(
            f"Missing required parameters: {', '.join(missing)}",
            tool_name=tool_name,
            details={"missing_params": missing},
        )


def safe_result(
    result: Any,
    error_message: str = "An error occurred",
    tool_name: str = "unknown",
) -> str:
    """
    Safely convert a result to a string, handling errors.
    
    Args:
        result: The result to convert
        error_message: Message to return on error
        tool_name: Name of the tool for logging
    
    Returns:
        String representation of the result
    """
    try:
        if result is None:
            return "No result"
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            import json
            return json.dumps(result, indent=2, default=str)
        return str(result)
    except Exception as e:
        logger.warning(
            "Failed to convert result to string",
            tool_name=tool_name,
            error=str(e),
        )
        return error_message
