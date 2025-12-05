#!/usr/bin/env python3
"""
Smoke test script for the MCP server.

This script tests basic server functionality and tool availability.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def run_smoke_tests():
    """Run smoke tests against the MCP server tools."""
    print("=" * 50)
    print("MCP Server Smoke Tests")
    print("=" * 50)
    print()
    
    # Import after path setup
    from src.config.settings import get_settings
    from src.common.logging_config import setup_logging
    
    setup_logging()
    settings = get_settings()
    
    print(f"Server Name: {settings.mcp_server_name}")
    print(f"Environment: {settings.environment}")
    print()
    
    results = {
        "passed": 0,
        "failed": 0,
        "tests": [],
    }
    
    def test_passed(name: str, message: str = ""):
        results["passed"] += 1
        results["tests"].append({"name": name, "status": "PASSED", "message": message})
        print(f"✅ {name}")
        if message:
            print(f"   {message}")
    
    def test_failed(name: str, error: str):
        results["failed"] += 1
        results["tests"].append({"name": name, "status": "FAILED", "error": error})
        print(f"❌ {name}")
        print(f"   Error: {error}")
    
    # Test 1: Settings load correctly
    print("\n--- Configuration Tests ---")
    try:
        assert settings.mcp_server_host is not None
        assert settings.mcp_server_port > 0
        test_passed("Settings load", f"Port: {settings.mcp_server_port}")
    except Exception as e:
        test_failed("Settings load", str(e))
    
    # Test 2: Tools can be imported
    print("\n--- Tool Import Tests ---")
    try:
        from src.tools.example_tools import register_example_tools
        test_passed("Example tools import")
    except Exception as e:
        test_failed("Example tools import", str(e))
    
    # Test 3: Test get_current_time tool
    print("\n--- Tool Execution Tests ---")
    try:
        from unittest.mock import MagicMock
        from src.tools.example_tools import register_example_tools
        
        # Capture tools
        registered_tools = {}
        
        def capture_tool():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp = MagicMock()
        mock_mcp.tool = capture_tool
        register_example_tools(mock_mcp)
        
        result = registered_tools["get_current_time"]()
        assert "-" in result and ":" in result
        test_passed("get_current_time", f"Result: {result}")
    except Exception as e:
        test_failed("get_current_time", str(e))
    
    # Test 4: Test calculate tool
    try:
        result = registered_tools["calculate"]("10 + 5 * 2")
        assert "20" in result
        test_passed("calculate", f"10 + 5 * 2 = {result}")
    except Exception as e:
        test_failed("calculate", str(e))
    
    # Test 5: Test echo_message tool
    try:
        result = registered_tools["echo_message"]("Test message", uppercase=True)
        assert "TEST MESSAGE" in result
        test_passed("echo_message", f"Result: {result}")
    except Exception as e:
        test_failed("echo_message", str(e))
    
    # Test 6: Test format_json tool
    try:
        test_json = '{"key": "value", "number": 42}'
        result = registered_tools["format_json"](test_json)
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        test_passed("format_json", "JSON formatted correctly")
    except Exception as e:
        test_failed("format_json", str(e))
    
    # Test 7: Test generate_uuid tool
    try:
        result = registered_tools["generate_uuid"]()
        assert len(result) == 36
        assert result.count("-") == 4
        test_passed("generate_uuid", f"UUID: {result}")
    except Exception as e:
        test_failed("generate_uuid", str(e))
    
    # Test 8: Test list_example_data tool
    try:
        result = registered_tools["list_example_data"](count=3, include_metadata=True)
        parsed = json.loads(result)
        assert len(parsed["items"]) == 3
        assert "metadata" in parsed
        test_passed("list_example_data", f"Returned {parsed['count']} items")
    except Exception as e:
        test_failed("list_example_data", str(e))
    
    # Test 9: Test server module import
    print("\n--- Server Module Tests ---")
    try:
        from src.server import mcp, health_check
        assert mcp is not None
        test_passed("Server module import")
    except Exception as e:
        test_failed("Server module import", str(e))
    
    # Test 10: Test health check
    try:
        from src.server import health_check
        result = await health_check()
        assert result["status"] == "healthy"
        test_passed("Health check", f"Status: {result['status']}")
    except Exception as e:
        test_failed("Health check", str(e))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Total:  {results['passed'] + results['failed']}")
    
    if results["failed"] > 0:
        print("\n⚠️  Some tests failed!")
        return 1
    else:
        print("\n🎉 All tests passed!")
        return 0


def main():
    """Main entry point."""
    exit_code = asyncio.run(run_smoke_tests())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
