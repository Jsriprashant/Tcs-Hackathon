"""Finance Agent graph for financial due diligence analysis."""

import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

from src.config.llm_config import get_llm
from src.finance_agent.state import FinanceAgentState
from src.finance_agent.tools import (
    analyze_balance_sheet,
    analyze_income_statement,
    analyze_cash_flow,
    analyze_financial_ratios,
    assess_financial_risk,
    compare_financial_performance,
)
from src.rag_agent.tools import retrieve_financial_documents
from src.common.logging_config import get_logger
from src.common.utils import invoke_llm_with_retry

logger = get_logger(__name__)

FINANCE_AGENT_PROMPT = """You are a Senior Financial Analyst Agent specialized in M&A Due Diligence.

Your role is to perform comprehensive financial analysis of target companies for mergers and acquisitions.

## Your Responsibilities:
1. Analyze financial statements (Income Statement, Balance Sheet, Cash Flow)
2. Calculate and interpret key financial ratios
3. Identify financial red flags and risks
4. Assess the financial health and sustainability of the target company
5. Provide valuation estimates and deal recommendations

## Analysis Framework:
1. **Profitability Analysis**: Margins, ROA, ROE
2. **Liquidity Analysis**: Current ratio, Quick ratio, Working capital
3. **Solvency Analysis**: Debt ratios, Interest coverage
4. **Growth Analysis**: Revenue trends, YoY growth
5. **Cash Flow Analysis**: Operating cash flow quality, Free cash flow

## Red Flags to Watch:
- Inconsistent revenue growth patterns
- Declining profit margins
- High or increasing debt levels
- Negative operating cash flow
- Significant discrepancies between net income and cash flow
- Unusual accounting adjustments

## Output Format:
- Provide clear, structured analysis
- Include specific numbers and ratios
- Highlight risks with severity ratings
- Give actionable recommendations
- Calculate an overall financial risk score (0-1)

First retrieve the relevant financial documents, then perform your analysis systematically.
"""

# Tools available to the finance agent
finance_tools = [
    retrieve_financial_documents,
    analyze_balance_sheet,
    analyze_income_statement,
    analyze_cash_flow,
    analyze_financial_ratios,
    assess_financial_risk,
    compare_financial_performance,
]


def create_finance_agent_node(state: FinanceAgentState) -> dict:
    """Finance agent node that performs financial analysis."""
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(finance_tools)
    
    messages = [SystemMessage(content=FINANCE_AGENT_PROMPT)] + state.messages
    
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
