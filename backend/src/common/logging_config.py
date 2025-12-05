"""Structured logging configuration for M&A Due Diligence Platform."""

import logging
import sys
from typing import Any, Optional
from datetime import datetime
import structlog
from functools import lru_cache


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@lru_cache()
def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def log_agent_action(
    logger: structlog.BoundLogger,
    agent_name: str,
    action: str,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Log an agent action with structured context."""
    logger.info(
        "agent_action",
        agent=agent_name,
        action=action,
        details=details or {},
        timestamp=datetime.now().isoformat(),
    )


def log_risk_assessment(
    logger: structlog.BoundLogger,
    category: str,
    score: float,
    factors: list[str],
) -> None:
    """Log a risk assessment result."""
    logger.info(
        "risk_assessment",
        category=category,
        score=score,
        factors=factors,
        timestamp=datetime.now().isoformat(),
    )


def log_tool_call(
    logger: structlog.BoundLogger,
    tool_name: str,
    params: dict[str, Any],
    result: Any = None,
    error: Optional[Exception] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """Log a tool invocation."""
    log_data = {
        "tool": tool_name,
        "params": params,
        "duration_ms": duration_ms,
        "timestamp": datetime.now().isoformat(),
    }
    
    if error:
        log_data["error"] = str(error)
        logger.error("tool_call_failed", **log_data)
    else:
        log_data["result_summary"] = str(result)[:200] if result else None
        logger.info("tool_call_success", **log_data)


def log_audit_event(
    logger: structlog.BoundLogger,
    event_type: str,
    user_id: Optional[str] = None,
    resource: Optional[str] = None,
    action: str = "",
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Log an audit event for compliance tracking."""
    logger.info(
        "audit_event",
        event_type=event_type,
        user_id=user_id,
        resource=resource,
        action=action,
        details=details or {},
        timestamp=datetime.now().isoformat(),
    )
