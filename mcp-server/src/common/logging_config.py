"""Logging configuration for MCP server."""

import logging
import sys
from typing import Any, Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    params: dict[str, Any],
    result: Any = None,
    error: Optional[Exception] = None,
    duration_ms: float = 0.0,
) -> None:
    """
    Log a tool call with consistent formatting.
    
    Args:
        logger: Logger instance
        tool_name: Name of the tool being called
        params: Input parameters
        result: Result of the call (if successful)
        error: Exception (if failed)
        duration_ms: Duration of the call in milliseconds
    """
    if error:
        logger.error(
            f"Tool '{tool_name}' failed after {duration_ms:.2f}ms",
            extra={
                "tool_name": tool_name,
                "params": params,
                "error": str(error),
                "duration_ms": duration_ms,
            }
        )
    else:
        logger.info(
            f"Tool '{tool_name}' completed in {duration_ms:.2f}ms",
            extra={
                "tool_name": tool_name,
                "params": params,
                "duration_ms": duration_ms,
            }
        )
