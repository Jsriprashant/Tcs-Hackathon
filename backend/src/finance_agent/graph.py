"""Finance Agent graph for TCS M&A Financial Due Diligence Analysis.

LLM-First Architecture:
- LLM handles reasoning, interpretation, and flexible data extraction
- Tools handle deterministic math calculations only
- 3 focused tools: get_financial_documents, calculate_ratios, calculate_tcs_score
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from src.config.llm_config import get_llm
from src.finance_agent.state import FinanceAgentState
from src.finance_agent.tools import finance_tools
from src.finance_agent.prompts import FINANCE_AGENT_SYSTEM_PROMPT
from src.common.logging_config import get_logger
from src.common.utils import invoke_llm_with_retry

logger = get_logger(__name__)


def create_finance_agent_node(state: FinanceAgentState) -> dict:
    """Finance agent node that performs financial analysis."""
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(finance_tools)
    
    messages = [SystemMessage(content=FINANCE_AGENT_SYSTEM_PROMPT)] + state.messages
    
    response, success = invoke_llm_with_retry(
        llm_with_tools, 
        messages,
        fallback_message="I encountered an issue performing financial analysis. The analysis service is temporarily unavailable. Please try again."
    )
    
    return {"messages": [response]}


def should_continue(state: FinanceAgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state.messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def build_finance_agent_graph() -> StateGraph:
    """Build the finance agent graph."""
    
    tool_node = ToolNode(finance_tools)
    
    workflow = StateGraph(FinanceAgentState)
    
    workflow.add_node("agent", create_finance_agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# Export compiled graph
finance_agent = build_finance_agent_graph()
graph = finance_agent  # Alias for langgraph.json
