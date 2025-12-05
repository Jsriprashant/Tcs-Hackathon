"""Example tools demonstrating MCP tool implementation patterns."""

import json
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.common.logging_config import get_logger
from src.common.errors import ToolError, ValidationError
from .base import tool_wrapper, validate_required_params, safe_result

logger = get_logger(__name__)


def register_example_tools(mcp: FastMCP) -> None:
    """
    Register example tools with the MCP server.
    
    These tools demonstrate common patterns for MCP tool implementation.
    Remove or replace these with your actual tools in production.
    
    Args:
        mcp: The FastMCP server instance
    """
    
    @mcp.tool()
    def get_current_time(
        format: str = "%Y-%m-%d %H:%M:%S",
        timezone: str = "UTC",
    ) -> str:
        """
        Get the current date and time.
        
        Args:
            format: DateTime format string (default: "%Y-%m-%d %H:%M:%S")
            timezone: Timezone name (default: "UTC") - currently only UTC supported
        
        Returns:
            Current datetime as formatted string
        """
        logger.info("get_current_time called", format=format, timezone=timezone)
        
        try:
            now = datetime.utcnow()
            return now.strftime(format)
        except Exception as e:
            logger.error("Failed to get current time", error=str(e))
            raise ToolError(
                f"Failed to format time: {e}",
                tool_name="get_current_time",
            )
    
    @mcp.tool()
    def calculate(expression: str) -> str:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
        
        Returns:
            Result of the calculation as a string
        """
        logger.info("calculate called", expression=expression)
        
        if not expression or not expression.strip():
            raise ValidationError(
                "Expression cannot be empty",
                field="expression",
            )
        
        # Only allow safe characters for math
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ToolError(
                "Invalid characters in expression. Only numbers and +,-,*,/,.,() are allowed.",
                tool_name="calculate",
                details={"expression": expression},
            )
        
        try:
            # Safe evaluation with restricted builtins
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            logger.error("Calculation failed", expression=expression, error=str(e))
            raise ToolError(
                f"Failed to evaluate expression: {e}",
                tool_name="calculate",
            )
    
    @mcp.tool()
    def echo_message(message: str, uppercase: bool = False) -> str:
        """
        Echo back a message (useful for testing).
        
        Args:
            message: The message to echo back
            uppercase: Whether to convert to uppercase (default: False)
        
        Returns:
            The echoed message
        """
        logger.info("echo_message called", message_length=len(message))
        
        if uppercase:
            message = message.upper()
        
        return f"Echo: {message}"
    
    @mcp.tool()
    def format_json(
        data: str,
        indent: int = 2,
    ) -> str:
        """
        Format a JSON string with proper indentation.
        
        Args:
            data: JSON string to format
            indent: Number of spaces for indentation (default: 2)
        
        Returns:
            Formatted JSON string
        """
        logger.info("format_json called", data_length=len(data), indent=indent)
        
        try:
            parsed = json.loads(data)
            formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
            return formatted
        except json.JSONDecodeError as e:
            logger.error("JSON parsing failed", error=str(e))
            raise ToolError(
                f"Invalid JSON: {e}",
                tool_name="format_json",
            )
    
    @mcp.tool()
    def generate_uuid() -> str:
        """
        Generate a new UUID.
        
        Returns:
            A new UUID string
        """
        import uuid
        
        new_uuid = str(uuid.uuid4())
        logger.info("generate_uuid called", uuid=new_uuid)
        return new_uuid
    
    @mcp.tool()
    def list_example_data(
        count: int = 5,
        include_metadata: bool = False,
    ) -> str:
        """
        Return example structured data (demonstrates returning complex data).
        
        Args:
            count: Number of items to return (1-100, default: 5)
            include_metadata: Whether to include metadata (default: False)
        
        Returns:
            JSON string with example data
        """
        logger.info("list_example_data called", count=count)
        
        # Validate count
        if count < 1:
            count = 1
        elif count > 100:
            count = 100
        
        items = [
            {
                "id": i,
                "name": f"Item {i}",
                "description": f"This is example item number {i}",
                "created_at": datetime.utcnow().isoformat(),
            }
            for i in range(1, count + 1)
        ]
        
        result: dict[str, Any] = {"items": items, "count": len(items)}
        
        if include_metadata:
            result["metadata"] = {
                "generated_at": datetime.utcnow().isoformat(),
                "version": "1.0",
            }
        
        return json.dumps(result, indent=2)
    
    logger.info("Example tools registered", tool_count=6)
