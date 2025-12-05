"""
Supervisor Agent Graph - Main orchestrator for M&A Due Diligence.

This module implements the main LangGraph workflow that coordinates
all specialized agents for comprehensive due diligence analysis.
"""

from typing import Literal, Any
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import time

from src.config.llm_config import get_llm
from src.supervisor.state import SupervisorState
from src.supervisor.prompts import SUPERVISOR_SYSTEM_PROMPT, ROUTING_PROMPT
from src.common.logging_config import get_logger, log_agent_action
from src.common.guardrails import PIIFilter, InputValidator

# Import sub-agents
from src.finance_agent.graph import finance_agent
from src.legal_agent.graph import legal_agent
from src.hr_agent.graph import hr_agent
from src.analyst_agent.graph import analyst_agent
from src.rag_agent.graph import rag_agent

logger = get_logger(__name__)

# Agent mapping
AGENTS = {
    "finance_agent": finance_agent,
    "legal_agent": legal_agent,
    "hr_agent": hr_agent,
    "analyst_agent": analyst_agent,
    "rag_agent": rag_agent,
}

# Greeting patterns for smart detection
GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "whats up", "sup", "yo",
    "hi there", "hello there", "hey there"
]

# Help/info request patterns
HELP_PATTERNS = [
    "help", "what can you do", "what do you do", "how can you help",
    "what are your capabilities", "what services", "tell me about yourself",
    "who are you", "what is this", "how does this work"
]


def is_greeting_or_simple_query(query: str) -> tuple[bool, str]:
    """
    Detect if the query is a simple greeting or help request.
    
    Returns:
        Tuple of (is_simple, response_type) where response_type is 'greeting', 'help', or 'none'
    """
    if not query:
        return False, "none"
    
    query_lower = query.lower().strip()
    
    # Check for greetings
    for pattern in GREETING_PATTERNS:
        if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.startswith(pattern + "!"):
            return True, "greeting"
    
    # Check for help requests
    for pattern in HELP_PATTERNS:
        if pattern in query_lower:
            return True, "help"
    
    # Check for very short queries that are likely greetings
    if len(query_lower.split()) <= 2 and any(g in query_lower for g in ["hi", "hello", "hey"]):
        return True, "greeting"
    
    return False, "none"


def get_greeting_response() -> str:
    """Get a friendly greeting response."""
    return """Hello! 👋 I'm your AI-powered M&A Due Diligence Assistant.

I can help you with comprehensive due diligence analysis for mergers and acquisitions, including:

📊 **Financial Analysis** - Analyze financial statements, calculate key ratios, identify red flags
⚖️ **Legal Review** - Review contracts, litigation history, IP portfolio, compliance issues
👥 **HR Assessment** - Evaluate workforce metrics, key person dependencies, cultural fit
📈 **Strategic Analysis** - Assess synergies, market position, deal recommendations

**To get started, you can:**
- Ask me to analyze a specific company
- Request a full due diligence report
- Ask specific questions about financial, legal, or HR aspects

How can I assist you with your M&A due diligence today?"""


def get_help_response() -> str:
    """Get a help/capabilities response."""
    return """# M&A Due Diligence Assistant - Capabilities

I'm an AI-powered platform designed to streamline M&A due diligence processes. Here's what I can do:

## 🔍 Available Analyses

### Financial Due Diligence
- Revenue and profitability analysis
- Cash flow assessment
- Debt and liability review
- Financial ratio calculations
- Red flag identification

### Legal Due Diligence
- Contract analysis (change of control provisions)
- Litigation history review
- IP portfolio assessment
- Regulatory compliance check

### HR Due Diligence
- Employee metrics analysis
- Attrition and retention assessment
- Key person dependency evaluation
- Cultural fit analysis

### Strategic Analysis
- Synergy estimation
- Market position analysis
- Deal structure recommendations
- Go/No-Go recommendations

## 💡 Example Queries

- "Analyze the financial health of TechCorp for acquisition"
- "What are the legal risks in acquiring StartupXYZ?"
- "Provide a full due diligence report for the merger between CompanyA and CompanyB"
- "What synergies can we expect from this horizontal merger?"

What would you like to analyze?"""


def create_supervisor_context(state: SupervisorState) -> str:
    """Create context string for supervisor prompt."""
    companies_context = ""
    if state.acquirer:
        companies_context += f"Acquirer: {state.acquirer.company_name} ({state.acquirer.industry})\n"
    if state.target:
        companies_context += f"Target: {state.target.company_name} ({state.target.industry})\n"
    if not companies_context:
        companies_context = "No companies specified yet."
    
    completed = ", ".join(state.agents_invoked) if state.agents_invoked else "None"
    
    return SUPERVISOR_SYSTEM_PROMPT.format(
        companies_context=companies_context,
        current_phase=state.current_phase,
        completed_analyses=completed,
    )


def supervisor_node(state: SupervisorState) -> dict:
    """
    Main supervisor node that decides which agent to invoke next.
    Handles greetings and simple queries without invoking the full agent flow.
    """
    llm = get_llm(temperature=0.0)
    
    # Validate input
    validator = InputValidator()
    if state.messages:
        last_message = state.messages[-1]
        if isinstance(last_message, HumanMessage):
            # Get the content as string
            content = last_message.content
            if isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)
            
            # Check for greetings or simple queries first
            is_simple, query_type = is_greeting_or_simple_query(content)
            if is_simple:
                if query_type == "greeting":
                    return {
                        "messages": [AIMessage(content=get_greeting_response())],
                        "next_agent": "FINISH",
                    }
                elif query_type == "help":
                    return {
                        "messages": [AIMessage(content=get_help_response())],
                        "next_agent": "FINISH",
                    }
            
            # Validate the query
            is_valid, error = validator.validate_query(content)
            if not is_valid:
                return {
                    "messages": [AIMessage(content=f"Invalid input: {error}")],
                    "next_agent": "FINISH",
                }
    
    # Filter PII from messages
    pii_filter = PIIFilter()
    filtered_messages = []
    for msg in state.messages:
        if hasattr(msg, 'content'):
            filtered_content, detected_pii = pii_filter.filter(msg.content)
            if detected_pii:
                logger.warning(f"PII detected and filtered: {detected_pii}")
            # Create new message with filtered content
            if isinstance(msg, HumanMessage):
                filtered_messages.append(HumanMessage(content=filtered_content))
            elif isinstance(msg, AIMessage):
                filtered_messages.append(AIMessage(content=filtered_content))
            else:
                filtered_messages.append(msg)
        else:
            filtered_messages.append(msg)
    
    # Build supervisor prompt
    system_prompt = create_supervisor_context(state)
    messages = [SystemMessage(content=system_prompt)] + filtered_messages
      # Get routing decision with retry logic for transient errors
    max_retries = 3
    retry_delay = 1  # seconds
    response = None
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            break
        except Exception as e:
            error_str = str(e)
            if "Internal Server Error" in error_str or "500" in error_str or "Connection" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM API error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s: {error_str}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"LLM API failed after {max_retries} attempts: {error_str}")
                    # Use default routing instead of failing
                    default_routing = determine_default_routing(state)
                    return {
                        "messages": [AIMessage(content=f"Using automated routing due to service issues. Proceeding with: {default_routing.get('next_agent', 'analysis')}")],
                        "next_agent": default_routing.get("next_agent", "FINISH"),
                    }
            else:
                logger.error(f"Unexpected LLM error: {error_str}")
                raise  # Re-raise non-transient errors
    
    if response is None:
        # Fallback to default routing
        default_routing = determine_default_routing(state)
        return {
            "messages": [AIMessage(content=f"Using automated routing. Proceeding with: {default_routing.get('next_agent', 'analysis')}")],
            "next_agent": default_routing.get("next_agent", "FINISH"),
        }
    
    log_agent_action(logger, "supervisor", "routing_decision", {"response": str(response.content)[:200]})
    
    # Parse routing decision
    try:
        # Try to extract JSON from response
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
            routing = json.loads(json_str)
        elif "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            routing = json.loads(content[start:end])
        else:
            # Default routing based on phase
            routing = determine_default_routing(state)
        
        next_agent = routing.get("next_agent", "FINISH")
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse routing: {e}")
        next_agent = determine_default_routing(state).get("next_agent", "FINISH")
    
    return {
        "messages": [response],
        "next_agent": next_agent,
    }


def determine_default_routing(state: SupervisorState) -> dict:
    """Determine default routing based on current phase."""
    
    if state.current_phase == "initialization":
        return {"next_agent": "rag_agent", "reasoning": "Start with document retrieval"}
    
    if "rag_agent" not in state.agents_invoked:
        return {"next_agent": "rag_agent", "reasoning": "Need to retrieve documents first"}
    
    if "finance_agent" not in state.agents_invoked:
        return {"next_agent": "finance_agent", "reasoning": "Perform financial analysis"}
    
    if "legal_agent" not in state.agents_invoked:
        return {"next_agent": "legal_agent", "reasoning": "Perform legal review"}
    
    if "hr_agent" not in state.agents_invoked:
        return {"next_agent": "hr_agent", "reasoning": "Perform HR assessment"}
    
    if "analyst_agent" not in state.agents_invoked:
        return {"next_agent": "analyst_agent", "reasoning": "Consolidate and recommend"}
    
    return {"next_agent": "FINISH", "reasoning": "All analyses complete"}


def finance_agent_node(state: SupervisorState) -> dict:
    """Invoke the finance agent."""
    log_agent_action(logger, "supervisor", "invoking_finance_agent", {})
    
    # Pass relevant state to finance agent
    result = finance_agent.invoke({
        "messages": state.messages,
        "deal_type": state.deal_type,
        "target_company": state.target,
    })
    
    return {
        "messages": result.get("messages", []),
        "agents_invoked": state.agents_invoked + ["finance_agent"],
        "current_phase": "legal_analysis",
    }


def legal_agent_node(state: SupervisorState) -> dict:
    """Invoke the legal agent."""
    log_agent_action(logger, "supervisor", "invoking_legal_agent", {})
    
    result = legal_agent.invoke({
        "messages": state.messages,
        "deal_type": state.deal_type,
        "target_company": state.target,
    })
    
    return {
        "messages": result.get("messages", []),
        "agents_invoked": state.agents_invoked + ["legal_agent"],
        "current_phase": "hr_analysis",
    }


def hr_agent_node(state: SupervisorState) -> dict:
    """Invoke the HR agent."""
    log_agent_action(logger, "supervisor", "invoking_hr_agent", {})
    
    result = hr_agent.invoke({
        "messages": state.messages,
        "deal_type": state.deal_type,
        "target_company": state.target,
        "acquirer_company": state.acquirer,
    })
    
    return {
        "messages": result.get("messages", []),
        "agents_invoked": state.agents_invoked + ["hr_agent"],
        "current_phase": "strategic_analysis",
    }


def analyst_agent_node(state: SupervisorState) -> dict:
    """Invoke the analyst agent for final consolidation."""
    log_agent_action(logger, "supervisor", "invoking_analyst_agent", {})
    
    # Convert risk_scores dict to list for AnalystAgentState compatibility
    risk_scores_list = list(state.risk_scores.values()) if isinstance(state.risk_scores, dict) else state.risk_scores
    
    result = analyst_agent.invoke({
        "messages": state.messages,
        "deal_type": state.deal_type,
        "analysis_type": state.analysis_type,
        "target_company": state.target,
        "acquirer_company": state.acquirer,
        "risk_scores": risk_scores_list,
    })
    
    return {
        "messages": result.get("messages", []),
        "agents_invoked": state.agents_invoked + ["analyst_agent"],
        "current_phase": "complete",
    }


def rag_agent_node(state: SupervisorState) -> dict:
    """Invoke the RAG agent for document retrieval."""
    log_agent_action(logger, "supervisor", "invoking_rag_agent", {})
    
    result = rag_agent.invoke({
        "messages": state.messages,
        "target_company": state.target,
    })
    
    return {
        "messages": result.get("messages", []),
        "agents_invoked": state.agents_invoked + ["rag_agent"],
        "current_phase": "financial_analysis",
    }


def human_review_node(state: SupervisorState) -> dict:
    """Node for human-in-the-loop review."""
    log_agent_action(logger, "supervisor", "requesting_human_review", {
        "reason": "High risk or critical decision required"
    })
    
    return {
        "pending_human_review": True,
        "messages": [AIMessage(content="""
## Human Review Required

This analysis requires human oversight before proceeding.

**Reason**: High risk factors identified or critical decision point reached.

Please review the findings and provide your feedback to continue.

**Options**:
1. Approve and continue with recommendations
2. Request additional analysis
3. Modify risk thresholds
4. Reject and terminate analysis
        """)],
    }


def route_to_agent(state: SupervisorState) -> str:
    """Route to the appropriate agent based on supervisor decision."""
    next_agent = state.next_agent
    
    if next_agent == "FINISH":
        return "end"
    elif next_agent == "human":
        return "human_review"
    elif next_agent in AGENTS:
        return next_agent
    else:
        # Default to supervisor for re-evaluation
        return "supervisor"


def build_supervisor_graph():
    """
    Build the main supervisor graph that orchestrates all agents.
    
    Graph Structure:
    
    START → supervisor → [routing decision]
                ↓
    ┌─────────────────────────────────────┐
    │  finance_agent  legal_agent         │
    │  hr_agent  analyst_agent  rag_agent │
    │  human_review                       │
    └─────────────────────────────────────┘
                ↓
            supervisor (loop back)
                ↓
              END
    """
    
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("finance_agent", finance_agent_node)
    workflow.add_node("legal_agent", legal_agent_node)
    workflow.add_node("hr_agent", hr_agent_node)
    workflow.add_node("analyst_agent", analyst_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("human_review", human_review_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "finance_agent": "finance_agent",
            "legal_agent": "legal_agent",
            "hr_agent": "hr_agent",
            "analyst_agent": "analyst_agent",
            "rag_agent": "rag_agent",
            "human_review": "human_review",
            "supervisor": "supervisor",
            "end": END,
        }
    )
    
    # All agents route back to supervisor
    for agent_name in ["finance_agent", "legal_agent", "hr_agent", "analyst_agent", "rag_agent"]:
        workflow.add_edge(agent_name, "supervisor")
    
    # Human review can go back to supervisor or end
    workflow.add_edge("human_review", END)
    
    return workflow.compile()


# Export the compiled graph for langgraph.json
graph = build_supervisor_graph()
