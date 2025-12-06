"""Custom error classes for MCP server."""

from typing import Optional, Any


class MCPServerError(Exception):
    """Base exception for MCP server errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ToolError(MCPServerError):
    """Error raised when a tool execution fails."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.tool_name = tool_name
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["tool_name"] = self.tool_name
        return result


class ValidationError(MCPServerError):
    """Error raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.field = field
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.field:
            result["field"] = self.field
        return result


class ExternalServiceError(MCPServerError):
    """Error raised when an external service call fails."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.service_name = service_name
        self.status_code = status_code
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["service_name"] = self.service_name
        if self.status_code:
            result["status_code"] = self.status_code
        return result
