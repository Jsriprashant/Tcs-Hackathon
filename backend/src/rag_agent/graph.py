# filepath: c:\Users\GenAIBLRANCUSR25.01HW2562306\Desktop\application_v1\Tcs-Hackathon\backend\src\rag_agent\graph.py
"""RAG Agent graph for document retrieval and search."""

import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

from src.config.llm_config import get_llm
from src.rag_agent.state import RAGAgentState
from src.rag_agent.tools import (
    retrieve_financial_documents,
    retrieve_legal_documents,
    retrieve_hr_documents,
    retrieve_employee_records,
    retrieve_contracts,
    retrieve_litigation_records,
    search_all_documents,
    get_company_overview,
)
from src.common.logging_config import get_logger
from src.common.utils import invoke_llm_with_retry

logger = get_logger(__name__)

RAG_AGENT_PROMPT = """You are a Document Retrieval Agent specialized in M&A Due Diligence.

Your role is to retrieve relevant documents from the document database to support due diligence analysis.

## Your Capabilities:
1. Retrieve financial documents (balance sheets, income statements, cash flows)
2. Retrieve legal documents (contracts, litigation, IP, compliance)
3. Retrieve HR documents (employee data, policies, handbooks)
4. Search across all document collections
5. Provide company overviews

## Available Companies:
- BBD (BBD Ltd)
- XYZ (XYZ Ltd)
- SUPERNOVA (Supernova Inc)
- RASPUTIN (Rasputin Petroleum Ltd)
- TECHNOBOX (Techno Box Inc)

## When to Use Each Tool:
- `retrieve_financial_documents`: For financial statements, balance sheets, revenue data
- `retrieve_legal_documents`: For contracts, legal compliance, litigation
- `retrieve_hr_documents`: For HR policies, employee handbooks
- `retrieve_employee_records`: For specific employee data, headcount, departments
- `retrieve_contracts`: For customer/vendor contracts, agreements
- `retrieve_litigation_records`: For lawsuits, court cases, legal disputes
- `search_all_documents`: For broad searches across all categories
- `get_company_overview`: For a summary of all available data for a company

## Output Format:
- Return the retrieved documents clearly formatted
- Indicate the source and type of each document
- Highlight relevant sections for the query
- Note if documents are not found for a query
"""

rag_tools = [
    retrieve_financial_documents,
    retrieve_legal_documents,
    retrieve_hr_documents,
    retrieve_employee_records,
    retrieve_contracts,
    retrieve_litigation_records,
    search_all_documents,
    get_company_overview,
]


def create_rag_agent_node(state: RAGAgentState) -> dict:
    """RAG agent node that retrieves documents."""
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(rag_tools)
    
    messages = [SystemMessage(content=RAG_AGENT_PROMPT)] + state.messages
    
    response, success = invoke_llm_with_retry(
        llm_with_tools,
        messages,
        fallback_message="I encountered an issue retrieving documents. The document search service is temporarily unavailable. Please try again."
    )
    
    return {"messages": [response]}


def should_continue(state: RAGAgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state.messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def build_rag_agent_graph() -> StateGraph:
    """Build the RAG agent graph."""
    
    tool_node = ToolNode(rag_tools)
    
    workflow = StateGraph(RAGAgentState)
    
    workflow.add_node("agent", create_rag_agent_node)
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


rag_agent = build_rag_agent_graph()
graph = rag_agent  # Alias for langgraph.json
