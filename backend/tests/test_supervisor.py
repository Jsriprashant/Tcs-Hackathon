"""Tests for the supervisor agent."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.supervisor.graph import create_supervisor_graph
from src.supervisor.nodes import router_node, route_to_agent
from src.supervisor.state import SupervisorState, AGENT_DESCRIPTIONS


class TestSupervisorGraph:
    """Test supervisor graph creation and compilation."""
    
    def test_create_supervisor_graph(self, checkpointer):
        """Test that supervisor graph can be created."""
        graph = create_supervisor_graph(checkpointer=checkpointer)
        assert graph is not None
    
    def test_supervisor_graph_has_nodes(self, checkpointer):
        """Test that supervisor graph has expected nodes."""
        graph = create_supervisor_graph(checkpointer=checkpointer)
        # Graph should have nodes defined
        assert graph is not None


class TestRouterNode:
    """Test the router node."""
    
    @pytest.mark.asyncio
    async def test_router_node_routes_to_agent(self, mock_llm):
        """Test that router correctly routes to an agent."""
        state: SupervisorState = {
            "messages": [{"role": "user", "content": "Give me an example"}],
            "next_agent": "",
            "agent_results": {},
            "routing_reason": "",
            "metadata": {},
        }
        
        with patch("src.supervisor.nodes.get_llm", return_value=mock_llm):
            mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
                content='{"next_agent": "example_agent", "routing_reason": "User wants an example"}'
            ))
            
            result = await router_node(state)
            
            assert result["next_agent"] == "example_agent"
            assert "routing_reason" in result
    
    @pytest.mark.asyncio
    async def test_router_node_handles_empty_messages(self):
        """Test router handles empty messages gracefully."""
        state: SupervisorState = {
            "messages": [],
            "next_agent": "",
            "agent_results": {},
            "routing_reason": "",
            "metadata": {},
        }
        
        result = await router_node(state)
        
        assert result["next_agent"] == "FINISH"


class TestRouteToAgent:
    """Test the route_to_agent conditional edge function."""
    
    def test_route_to_example_agent(self):
        """Test routing to example agent."""
        state: SupervisorState = {
            "messages": [],
            "next_agent": "example_agent",
            "agent_results": {},
            "routing_reason": "",
            "metadata": {},
        }
        
        result = route_to_agent(state)
        assert result == "example_agent"
    
    def test_route_to_finish(self):
        """Test routing to finish."""
        state: SupervisorState = {
            "messages": [],
            "next_agent": "FINISH",
            "agent_results": {},
            "routing_reason": "",
            "metadata": {},
        }
        
        result = route_to_agent(state)
        assert result == "finish"
    
    def test_route_unknown_agent_fallback(self):
        """Test that unknown agent falls back to finish."""
        state: SupervisorState = {
            "messages": [],
            "next_agent": "unknown_agent",
            "agent_results": {},
            "routing_reason": "",
            "metadata": {},
        }
        
        result = route_to_agent(state)
        assert result == "finish"


class TestAgentDescriptions:
    """Test agent descriptions configuration."""
    
    def test_agent_descriptions_exist(self):
        """Test that agent descriptions are defined."""
        assert AGENT_DESCRIPTIONS is not None
        assert len(AGENT_DESCRIPTIONS) > 0
    
    def test_example_agent_has_description(self):
        """Test that example agent has a description."""
        assert "example_agent" in AGENT_DESCRIPTIONS
        assert len(AGENT_DESCRIPTIONS["example_agent"]) > 0
