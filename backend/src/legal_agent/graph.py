"""Legal Agent graph for legal due diligence analysis."""

import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

from src.config.llm_config import get_llm
from src.legal_agent.state import LegalAgentState
from src.legal_agent.tools import (
    analyze_contracts,
    analyze_litigation_exposure,
    analyze_ip_portfolio,
    analyze_regulatory_compliance,
    analyze_corporate_governance,
    generate_legal_risk_score,
)
from src.rag_agent.tools import retrieve_legal_documents
from src.common.logging_config import get_logger
from src.common.utils import invoke_llm_with_retry

logger = get_logger(__name__)

LEGAL_AGENT_PROMPT = """You are a Senior Legal Analyst Agent specialized in M&A Due Diligence.

Your role is to perform comprehensive legal analysis of target companies for mergers and acquisitions.

## Your Responsibilities:
1. Analyze litigation history and pending cases
2. Review material contracts for change-of-control provisions
3. Assess intellectual property portfolio and risks
4. Evaluate regulatory compliance status
5. Identify legal red flags and deal-breakers

## Analysis Framework:

### 1. Litigation Analysis
- Review pending and historical litigation
- Assess potential liability exposure
- Identify patterns of disputes

### 2. Contract Analysis
- Change of control provisions
- Termination rights
- Material contract dependencies
- Assignment restrictions

### 3. IP Due Diligence
- Patent portfolio review
- Trademark and copyright status
- Freedom to operate assessment
- IP disputes and claims

### 4. Compliance Review
- Regulatory compliance status
- Historical violations
- Pending investigations
- Required licenses and permits

## Red Flags to Watch:
- Multiple pending lawsuits
- Class action exposure
- Material contracts with termination on CoC
- IP disputes or challenges
- Regulatory investigations
- Pattern of compliance violations

## Output Format:
- Provide structured legal analysis
- Quantify potential liabilities where possible
- Rate severity of each risk area
- Give clear recommendations for deal structure
- Calculate overall legal risk score (0-1)

First retrieve the relevant legal documents, then perform your analysis systematically.
"""

legal_tools = [
    retrieve_legal_documents,
    analyze_contracts,
    analyze_litigation_exposure,
    analyze_ip_portfolio,
    analyze_regulatory_compliance,
    analyze_corporate_governance,
    generate_legal_risk_score,
]


def create_legal_agent_node(state: LegalAgentState) -> dict:
    """Legal agent node that performs legal due diligence."""
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(legal_tools)
    
    messages = [SystemMessage(content=LEGAL_AGENT_PROMPT)] + state.messages
    
    response, success = invoke_llm_with_retry(
        llm_with_tools,
        messages,
        fallback_message="I encountered an issue performing legal analysis. The analysis service is temporarily unavailable. Please try again."
    )
    
    return {"messages": [response]}


def should_continue(state: LegalAgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state.messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def build_legal_agent_graph() -> StateGraph:
    """Build the legal agent graph."""
    
    tool_node = ToolNode(legal_tools)
    
    workflow = StateGraph(LegalAgentState)
    
    workflow.add_node("agent", create_legal_agent_node)
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


legal_agent = build_legal_agent_graph()
graph = legal_agent  # Alias for langgraph.json
