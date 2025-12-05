"""Pytest configuration and fixtures."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def checkpointer():
    """Provide an in-memory checkpointer for tests."""
    return MemorySaver()


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_state(sample_messages):
    """Provide a sample agent state for testing."""
    return {
        "messages": sample_messages,
        "metadata": {"test": True},
        "is_last_step": False,
    }


@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(
        content="This is a test response.",
        tool_calls=[],
    ))
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


@pytest.fixture
def mock_mcp_client():
    """Provide a mock MCP client for testing."""
    mock = AsyncMock()
    mock.call_tool = AsyncMock(return_value={"success": True, "result": "test"})
    mock.list_tools = AsyncMock(return_value=[
        {"name": "test_tool", "description": "A test tool"},
    ])
    return mock


@pytest.fixture
def thread_config():
    """Provide a thread configuration for testing."""
    return {
        "configurable": {
            "thread_id": "test-thread-001",
        }
    }
