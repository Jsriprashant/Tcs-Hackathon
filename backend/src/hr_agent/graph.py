"""HR Agent graph for M&A HR Policy Comparison - Acquirer Focus."""

import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

from src.config.llm_config import get_llm
from src.hr_agent.state import HRAgentState
from src.hr_agent.prompts import HR_AGENT_SYSTEM_PROMPT, HR_AGENT_COMPACT_PROMPT
from src.hr_agent.tools import (
    # PRIMARY: Smart Policy Comparison Tools
    get_acquirer_baseline,
    get_target_hr_policies,
    compare_policy_category,
    calculate_hr_compatibility_score,
    get_scoring_rubrics,
    check_deal_breakers,
    get_integration_effort_estimate,
    # LEGACY: Kept for backward compatibility
    analyze_employee_data,
    analyze_attrition,
    analyze_key_person_dependency,
    analyze_hr_policies,
    analyze_hr_compliance,
    analyze_culture_fit,
    generate_hr_risk_score,
)
from src.common.logging_config import get_logger
from src.common.utils import invoke_llm_with_retry

logger = get_logger(__name__)

# Use the new policy comparison prompt
HR_AGENT_PROMPT = HR_AGENT_SYSTEM_PROMPT  # From prompts.py

# Define tools for HR Agent (prioritize policy comparison tools)
hr_tools = [
    # PRIMARY: Smart Policy Comparison Tools
    get_acquirer_baseline,
    get_target_hr_policies,
    compare_policy_category,
    calculate_hr_compatibility_score,
    get_scoring_rubrics,
    check_deal_breakers,
    get_integration_effort_estimate,
    # SECONDARY: Legacy tools (if needed)
    analyze_employee_data,
    analyze_attrition,
    analyze_key_person_dependency,
    analyze_hr_policies,
    analyze_hr_compliance,
    analyze_culture_fit,
    generate_hr_risk_score,
]


def create_hr_agent_node(state: HRAgentState) -> dict:
    """HR agent node that performs HR due diligence."""
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(hr_tools)
    
    messages = [SystemMessage(content=HR_AGENT_PROMPT)] + state.messages
    
    response, success = invoke_llm_with_retry(
        llm_with_tools,
        messages,
        fallback_message="I encountered an issue performing HR analysis. The analysis service is temporarily unavailable. Please try again."
    )
    
    return {"messages": [response]}


def should_continue(state: HRAgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state.messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def build_hr_agent_graph() -> StateGraph:
    """Build the HR agent graph."""
    
    tool_node = ToolNode(hr_tools)
    
    workflow = StateGraph(HRAgentState)
    
    workflow.add_node("agent", create_hr_agent_node)
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


hr_agent = build_hr_agent_graph()
graph = hr_agent  # Alias for langgraph.json
