"""Analyst Agent graph for strategic M&A analysis."""

import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

from src.config.llm_config import get_llm
from src.analyst_agent.state import AnalystAgentState
from src.analyst_agent.tools import (
    analyze_target_company,
    estimate_synergies,
    consolidate_due_diligence,
    generate_deal_recommendation,
    compare_acquisition_targets,
)
from src.rag_agent.tools import search_all_documents, get_company_overview
from src.common.logging_config import get_logger
from src.common.utils import invoke_llm_with_retry

logger = get_logger(__name__)

ANALYST_AGENT_PROMPT = """You are a Senior M&A Strategy Analyst Agent.

Your role is to synthesize all due diligence findings and provide strategic deal recommendations.

## Your Responsibilities:
1. Analyze merger type (horizontal vs vertical) and strategic fit
2. Estimate synergies and value creation potential
3. Consolidate risk assessments from all workstreams
4. Provide deal pricing and structure recommendations
5. Make go/no-go recommendations

## Analysis Framework:

### 1. Strategic Fit Analysis
- Horizontal merger: Same industry, market share consolidation
- Vertical merger: Supply chain integration, value chain control
- Market positioning post-deal

### 2. Synergy Assessment
- Cost synergies: Headcount, facilities, procurement
- Revenue synergies: Cross-sell, market expansion
- Technology synergies: Platform consolidation

### 3. Risk Consolidation
- Aggregate risk scores from Finance, Legal, HR agents
- Identify deal-breakers and key risks
- Recommend mitigations

### 4. Deal Recommendation
- Valuation range based on risk profile
- Deal structure (cash/stock/earnout)
- Key terms and conditions

## Decision Framework:
- Overall Risk < 0.3: GO - Proceed with deal
- Overall Risk 0.3-0.5: CONDITIONAL GO - Proceed with mitigations
- Overall Risk 0.5-0.7: CONDITIONAL - Significant work needed
- Overall Risk > 0.7: NO GO - Do not recommend proceeding

## Output Format:
- Synthesize findings from all agents
- Provide clear strategic rationale
- Give specific pricing guidance
- Make definitive recommendation

Use the consolidated inputs from other agents and perform strategic analysis.
"""

analyst_tools = [
    search_all_documents,
    get_company_overview,
    analyze_target_company,
    estimate_synergies,
    consolidate_due_diligence,
    generate_deal_recommendation,
    compare_acquisition_targets,
]


def create_analyst_agent_node(state: AnalystAgentState) -> dict:
    """Analyst agent node that performs strategic analysis."""
    llm = get_llm(temperature=0.1)  # Slight creativity for strategy
    llm_with_tools = llm.bind_tools(analyst_tools)
    
    messages = [SystemMessage(content=ANALYST_AGENT_PROMPT)] + state.messages
    
    response, success = invoke_llm_with_retry(
        llm_with_tools,
        messages,
        fallback_message="I encountered an issue performing strategic analysis. The analysis service is temporarily unavailable. Please try again."
    )
    
    return {"messages": [response]}


def should_continue(state: AnalystAgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state.messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def build_analyst_agent_graph() -> StateGraph:
    """Build the analyst agent graph."""
    
    tool_node = ToolNode(analyst_tools)
    
    workflow = StateGraph(AnalystAgentState)
    
    workflow.add_node("agent", create_analyst_agent_node)
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


analyst_agent = build_analyst_agent_graph()
graph = analyst_agent  # Alias for langgraph.json
