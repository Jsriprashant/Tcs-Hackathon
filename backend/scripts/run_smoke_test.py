#!/usr/bin/env python3
"""
Smoke test script for the backend skeleton.

This script performs basic validation that the agent system is working correctly.
It tests:
1. Agent graph creation
2. Basic message processing
3. Tool execution
4. MCP client connectivity (if configured)

Usage:
    python scripts/run_smoke_test.py
    python scripts/run_smoke_test.py --verbose
    python scripts/run_smoke_test.py --include-mcp
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage
from src.common.logging_config import get_logger
from src.common.checkpointers import get_checkpointer
from src.common.store import get_store
from src.config.settings import get_settings

logger = get_logger(__name__)


class SmokeTestResult:
    """Result of a smoke test."""
    
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f" - {self.message}" if self.message else ""
        return f"{status}: {self.name}{msg}"


async def test_supervisor_creation() -> SmokeTestResult:
    """Test that the supervisor graph can be created."""
    try:
        from src.supervisor.graph import create_supervisor_graph
        
        checkpointer = get_checkpointer()
        store = get_store()
        graph = create_supervisor_graph(checkpointer=checkpointer, store=store)
        
        if graph is not None:
            return SmokeTestResult("Supervisor Creation", True)
        else:
            return SmokeTestResult("Supervisor Creation", False, "Graph is None")
    except Exception as e:
        return SmokeTestResult("Supervisor Creation", False, str(e))


async def test_example_agent_creation() -> SmokeTestResult:
    """Test that the example agent can be created."""
    try:
        from src.example_agent.graph import create_example_agent
        
        checkpointer = get_checkpointer()
        agent = create_example_agent(checkpointer=checkpointer)
        
        if agent is not None:
            return SmokeTestResult("Example Agent Creation", True)
        else:
            return SmokeTestResult("Example Agent Creation", False, "Agent is None")
    except Exception as e:
        return SmokeTestResult("Example Agent Creation", False, str(e))


async def test_example_agent_tools() -> SmokeTestResult:
    """Test that example agent tools work."""
    try:
        from src.example_agent.tools import get_current_time, calculate, echo_message
        
        # Test get_current_time
        time_result = get_current_time.invoke({})
        if not time_result:
            return SmokeTestResult("Agent Tools", False, "get_current_time returned empty")
        
        # Test calculate
        calc_result = calculate.invoke({"expression": "2 + 2"})
        if "4" not in calc_result:
            return SmokeTestResult("Agent Tools", False, f"calculate returned unexpected: {calc_result}")
        
        # Test echo_message
        echo_result = echo_message.invoke({"message": "test"})
        if "test" not in echo_result:
            return SmokeTestResult("Agent Tools", False, f"echo_message returned unexpected: {echo_result}")
        
        return SmokeTestResult("Agent Tools", True, "All tools working")
    except Exception as e:
        return SmokeTestResult("Agent Tools", False, str(e))


async def test_checkpointer() -> SmokeTestResult:
    """Test that checkpointer can be created."""
    try:
        checkpointer = get_checkpointer()
        
        if checkpointer is not None:
            checkpointer_type = type(checkpointer).__name__
            return SmokeTestResult("Checkpointer", True, f"Using {checkpointer_type}")
        else:
            return SmokeTestResult("Checkpointer", False, "Checkpointer is None")
    except Exception as e:
        return SmokeTestResult("Checkpointer", False, str(e))


async def test_store() -> SmokeTestResult:
    """Test that store can be created."""
    try:
        store = get_store()
        
        if store is not None:
            store_type = type(store).__name__
            return SmokeTestResult("Store", True, f"Using {store_type}")
        else:
            return SmokeTestResult("Store", True, "No store configured (optional)")
    except Exception as e:
        return SmokeTestResult("Store", False, str(e))


async def test_settings() -> SmokeTestResult:
    """Test that settings can be loaded."""
    try:
        settings = get_settings()
        
        if settings is not None:
            return SmokeTestResult("Settings", True, f"Environment: {settings.ENVIRONMENT}")
        else:
            return SmokeTestResult("Settings", False, "Settings is None")
    except Exception as e:
        return SmokeTestResult("Settings", False, str(e))


async def test_mcp_client() -> SmokeTestResult:
    """Test MCP client connectivity (if configured)."""
    try:
        settings = get_settings()
        
        if not settings.MCP_SERVER_URL:
            return SmokeTestResult("MCP Client", True, "MCP not configured (skipped)")
        
        from src.common.mcp_client import MCPClient
        
        async with MCPClient(base_url=settings.MCP_SERVER_URL) as client:
            tools = await client.list_tools()
            tool_count = len(tools) if tools else 0
            return SmokeTestResult("MCP Client", True, f"Connected, {tool_count} tools available")
    except Exception as e:
        return SmokeTestResult("MCP Client", False, str(e))


async def run_smoke_tests(
    verbose: bool = False,
    include_mcp: bool = False,
) -> tuple[list[SmokeTestResult], bool]:
    """Run all smoke tests."""
    results = []
    
    # Core tests
    tests = [
        ("Settings", test_settings),
        ("Checkpointer", test_checkpointer),
        ("Store", test_store),
        ("Supervisor Creation", test_supervisor_creation),
        ("Example Agent Creation", test_example_agent_creation),
        ("Agent Tools", test_example_agent_tools),
    ]
    
    if include_mcp:
        tests.append(("MCP Client", test_mcp_client))
    
    for test_name, test_func in tests:
        if verbose:
            print(f"Running: {test_name}...", end=" ", flush=True)
        
        result = await test_func()
        results.append(result)
        
        if verbose:
            print(result)
    
    all_passed = all(r.passed for r in results)
    return results, all_passed


def print_summary(results: list[SmokeTestResult], all_passed: bool):
    """Print test summary."""
    print("\n" + "=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    for result in results:
        print(result)
    
    print("-" * 50)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    if all_passed:
        print("\n🎉 All smoke tests passed!")
    else:
        print("\n⚠️  Some smoke tests failed. Check the output above.")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run smoke tests for the backend skeleton")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output during test execution",
    )
    parser.add_argument(
        "--include-mcp",
        action="store_true",
        help="Include MCP client connectivity test",
    )
    
    args = parser.parse_args()
    
    print("🔍 Running smoke tests for backend skeleton...")
    print()
    
    results, all_passed = await run_smoke_tests(
        verbose=args.verbose,
        include_mcp=args.include_mcp,
    )
    
    print_summary(results, all_passed)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
