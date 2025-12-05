"""Custom exceptions for M&A Due Diligence Platform."""

from typing import Optional, Any


class DueDiligenceError(Exception):
    """Base exception for all due diligence errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "DD_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class AgentError(DueDiligenceError):
    """Error occurring within an agent's execution."""
    
    def __init__(
        self, 
        agent_name: str, 
        message: str,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Agent '{agent_name}' error: {message}",
            error_code="AGENT_ERROR",
            details={"agent_name": agent_name, **(details or {})}
        )
        self.agent_name = agent_name


class DataNotFoundError(DueDiligenceError):
    """Requested data not found in the system."""
    
    def __init__(
        self, 
        data_type: str, 
        identifier: str,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(
            message=f"{data_type} not found for identifier: {identifier}",
            error_code="DATA_NOT_FOUND",
            details={"data_type": data_type, "identifier": identifier, **(details or {})}
        )


class ValidationError(DueDiligenceError):
    """Input validation error."""
    
    def __init__(
        self, 
        field: str, 
        message: str,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Validation error for '{field}': {message}",
            error_code="VALIDATION_ERROR",
            details={"field": field, **(details or {})}
        )


class RiskThresholdExceededError(DueDiligenceError):
    """Risk score exceeds acceptable threshold."""
    
    def __init__(
        self, 
        category: str, 
        score: float, 
        threshold: float,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Risk threshold exceeded for {category}: {score:.2f} > {threshold:.2f}",
            error_code="RISK_THRESHOLD_EXCEEDED",
            details={
                "category": category, 
                "score": score, 
                "threshold": threshold,
                **(details or {})
            }
        )


class ExternalServiceError(DueDiligenceError):
    """Error from external service (LLM, database, etc.)."""
    
    def __init__(
        self, 
        service_name: str, 
        message: str,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(
            message=f"External service '{service_name}' error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service_name": service_name, **(details or {})}
        )


class GuardrailViolationError(DueDiligenceError):
    """Guardrail or security policy violation."""
    
    def __init__(
        self, 
        guardrail_type: str, 
        message: str,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Guardrail violation ({guardrail_type}): {message}",
            error_code="GUARDRAIL_VIOLATION",
            details={"guardrail_type": guardrail_type, **(details or {})}
        )
