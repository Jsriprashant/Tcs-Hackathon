"""HR Agent graph for HR due diligence analysis."""

import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

from src.config.llm_config import get_llm
from src.hr_agent.state import HRAgentState
from src.hr_agent.tools import (
    analyze_employee_data,
    analyze_attrition,
    analyze_key_person_dependency,
    analyze_hr_policies,
    analyze_hr_compliance,
    analyze_culture_fit,
    generate_hr_risk_score,
)
from src.rag_agent.tools import retrieve_hr_documents, retrieve_employee_records
from src.common.logging_config import get_logger
from src.common.utils import invoke_llm_with_retry

logger = get_logger(__name__)

HR_AGENT_PROMPT = """You are a Senior HR Analyst Agent specialized in M&A Due Diligence.

Your role is to assess human capital risks and cultural compatibility for mergers and acquisitions.

## Your Responsibilities:
1. Analyze employee metrics and attrition patterns
2. Assess key person dependencies and succession risks
3. Review HR compliance and employment practices
4. Evaluate cultural fit and integration challenges
5. Review HR policies and identify gaps

## Analysis Framework:

### 1. Workforce Analysis
- Total headcount and composition
- Attrition rates vs industry benchmarks
- Employee satisfaction metrics

### 2. Key Person Risk
- Critical role identification
- Succession planning status
- Retention risk assessment

### 3. HR Compliance
- Employment disputes and claims
- Regulatory compliance status
- Union/CBA considerations

### 4. Culture Assessment
- Cultural compatibility analysis
- Work style alignment
- Integration complexity

### 5. Policy Review
- HR policy completeness
- Gap analysis
- Harmonization needs

## Red Flags to Watch:
- High attrition rates (>20% above benchmark)
- Multiple key person departures
- Discrimination claims or EEOC complaints
- Low employee satisfaction (<60%)
- Missing critical HR policies
- Significant culture mismatch

## Output Format:
- Provide structured HR analysis
- Quantify people risks where possible
- Rate severity of each risk area
- Give integration recommendations
- Calculate overall HR risk score (0-1)

First retrieve the relevant HR documents, then perform your analysis systematically.
"""

hr_tools = [
    retrieve_hr_documents,
    retrieve_employee_records,
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
