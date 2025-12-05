"""Tools module - register all MCP tools here."""

from mcp.server.fastmcp import FastMCP

from .example_tools import register_example_tools

# Import your custom tool modules here
# from .my_tools import register_my_tools


def register_all_tools(mcp: FastMCP) -> None:
    """
    Register all tools with the MCP server.
    
    Add new tool registration functions here when you create new tool modules.
    
    Args:
        mcp: The FastMCP server instance
    """
    # Register example tools (remove in production if not needed)
    register_example_tools(mcp)
    
    # Register your custom tools here
    # register_my_tools(mcp)
