"""Tests for the example agent."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.example_agent.graph import create_example_agent, create_example_agent_node
from src.example_agent.nodes import call_model, should_continue
from src.example_agent.tools import AGENT_TOOLS, get_current_time, calculate, echo_message


class TestExampleAgentGraph:
    """Test example agent graph creation."""
    
    def test_create_example_agent(self, checkpointer):
        """Test that example agent graph can be created."""
        agent = create_example_agent(checkpointer=checkpointer)
        assert agent is not None
    
    def test_create_example_agent_node(self):
        """Test that example agent node can be created for supervisor."""
        node = create_example_agent_node()
        assert node is not None
        assert callable(node)


class TestExampleAgentNodes:
    """Test example agent node functions."""
    
    @pytest.mark.asyncio
    async def test_call_model_returns_response(self, sample_state, mock_llm):
        """Test that call_model returns a response."""
        with patch("src.example_agent.nodes.get_llm", return_value=mock_llm):
            result = await call_model(sample_state)
            
            assert "messages" in result
            assert len(result["messages"]) > 0
    
    def test_should_continue_returns_end_when_no_tool_calls(self, sample_state):
        """Test should_continue returns 'end' when no tool calls."""
        from langchain_core.messages import AIMessage
        
        sample_state["messages"].append(AIMessage(content="Response", tool_calls=[]))
        
        result = should_continue(sample_state)
        assert result == "end"
    
    def test_should_continue_returns_tools_when_tool_calls_present(self, sample_state):
        """Test should_continue returns 'tools' when tool calls present."""
        from langchain_core.messages import AIMessage
        
        sample_state["messages"].append(AIMessage(
            content="",
            tool_calls=[{"id": "1", "name": "test", "args": {}}]
        ))
        
        result = should_continue(sample_state)
        assert result == "tools"


class TestExampleAgentTools:
    """Test example agent tools."""
    
    def test_tools_are_defined(self):
        """Test that agent tools are defined."""
        assert AGENT_TOOLS is not None
        assert len(AGENT_TOOLS) > 0
    
    def test_get_current_time(self):
        """Test get_current_time tool."""
        result = get_current_time.invoke({})
        assert result is not None
        assert len(result) > 0
        # Should be in YYYY-MM-DD HH:MM:SS format
        assert "-" in result
        assert ":" in result
    
    def test_calculate_addition(self):
        """Test calculate tool with addition."""
        result = calculate.invoke({"expression": "2 + 2"})
        assert "4" in result
    
    def test_calculate_multiplication(self):
        """Test calculate tool with multiplication."""
        result = calculate.invoke({"expression": "5 * 3"})
        assert "15" in result
    
    def test_calculate_invalid_expression(self):
        """Test calculate tool with invalid expression."""
        result = calculate.invoke({"expression": "import os"})
        assert "Error" in result
    
    def test_echo_message(self):
        """Test echo_message tool."""
        result = echo_message.invoke({"message": "Hello!"})
        assert "Hello!" in result
        assert "Echo" in result


class TestExampleAgentIntegration:
    """Integration tests for example agent."""
    
    @pytest.mark.asyncio
    async def test_agent_responds_to_simple_message(self, checkpointer, thread_config, mock_llm):
        """Test that agent can respond to a simple message."""
        with patch("src.example_agent.nodes.get_llm", return_value=mock_llm):
            agent = create_example_agent(checkpointer=checkpointer)
            
            result = await agent.ainvoke(
                {
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "metadata": {},
                    "is_last_step": False,
                },
                config=thread_config,
            )
            
            assert "messages" in result
            assert len(result["messages"]) > 0
