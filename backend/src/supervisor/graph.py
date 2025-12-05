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
from src.common.state import CompanyInfo

# Import intent classifier (Phase 1)
from src.common.intent_classifier import (
    classify_intent,
    get_last_human_message,
    IntentType,
    IntentClassificationResult,
)

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


# =============================================================================
# INTENT CLASSIFICATION NODE (Phase 3)
# This node runs FIRST to classify user intent and gate chain activation
# =============================================================================

def intent_classifier_node(state: SupervisorState) -> dict:
    """
    Classifies user intent and extracts company names.
    
    This node runs first, before any routing decision. It determines whether
    the full agent chain should be activated based on:
    1. Intent type (must be MA_DUE_DILIGENCE)
    2. Company names being present (acquirer and/or target)
    
    Args:
        state: Current supervisor state
        
    Returns:
        Dict with intent classification results and optional company info
    """
    log_agent_action(logger, "intent_classifier", "classifying_intent", {})
    
    # Get last user message
    last_message = get_last_human_message(state.messages)
    
    if not last_message:
        logger.warning("No human message found for intent classification")
        return {
            "intent_classified": True,
            "intent_type": "UNKNOWN",
            "intent_confidence": 0.0,
            "chain_activated": False,
        }
    
    # Classify intent using the intent classifier module
    result: IntentClassificationResult = classify_intent(last_message)
    
    log_agent_action(logger, "intent_classifier", "classification_result", {
        "intent": result.intent.value,
        "confidence": result.confidence,
        "acquirer": result.acquirer_company,
        "target": result.target_company,
        "should_activate_chain": result.should_activate_chain,
    })
    
    # Build state updates
    updates = {
        "intent_classified": True,
        "intent_type": result.intent.value,
        "intent_confidence": result.confidence,
        "chain_activated": result.should_activate_chain,
    }
    
    # Extract and set companies if M&A due diligence with company names
    if result.intent == IntentType.MA_DUE_DILIGENCE and result.should_activate_chain:
        if result.acquirer_company:
            updates["acquirer"] = CompanyInfo(
                company_id=result.acquirer_company.lower().replace(" ", "_"),
                company_name=result.acquirer_company,
                industry="Unknown"  # Will be populated by RAG agent
            )
            logger.info(f"Set acquirer company: {result.acquirer_company}")
        
        if result.target_company:
            updates["target"] = CompanyInfo(
                company_id=result.target_company.lower().replace(" ", "_"),
                company_name=result.target_company,
                industry="Unknown"  # Will be populated by RAG agent
            )
            logger.info(f"Set target company: {result.target_company}")
    
    logger.info(
        f"Intent classification complete: {result.intent.value}, "
        f"chain_activated={result.should_activate_chain}"
    )
    
    return updates


def route_after_intent(state: SupervisorState) -> str:
    """
    Route based on classified intent.
    
    This function is called after intent_classifier_node to determine
    the next node in the graph based on the classified intent.
    
    Only routes to supervisor (and agent chain) for MA_DUE_DILIGENCE with companies.
    All other intents route to specialized handlers that respond directly.
    
    Args:
        state: Current supervisor state with intent classification
        
    Returns:
        String name of the next node to execute
    """
    intent = state.intent_type
    
    log_agent_action(logger, "intent_router", "routing_decision", {
        "intent": intent,
        "chain_activated": state.chain_activated,
    })
    
    # MA_DUE_DILIGENCE with companies -> full agent chain via supervisor
    if intent == "MA_DUE_DILIGENCE" and state.chain_activated:
        logger.info("Routing to supervisor for full agent chain execution")
        return "supervisor"
    
    # MA_QUESTION -> answer M&A question directly (no chain)
    elif intent == "MA_QUESTION":
        logger.info("Routing to ma_question_handler")
        return "ma_question_handler"
    
    # GREETING -> greeting response (no chain)
    elif intent == "GREETING":
        logger.info("Routing to greeting_handler")
        return "greeting_handler"
    
    # HELP -> help/capabilities response (no chain)
    elif intent == "HELP":
        logger.info("Routing to help_handler")
        return "help_handler"
    
    # INFORMATIONAL or UNKNOWN -> informational response (no chain)
    else:
        logger.info(f"Routing to informational_handler for intent: {intent}")
        return "informational_handler"


# =============================================================================
# HANDLER NODES FOR NON-CHAIN QUERIES (Phase 5)
# These nodes handle queries that don't require the full agent chain
# =============================================================================

def greeting_handler_node(state: SupervisorState) -> dict:
    """
    Handle greeting queries with a friendly response.
    
    This node responds to simple greetings without invoking any agents.
    Uses the pre-defined greeting response for consistency.
    """
    log_agent_action(logger, "greeting_handler", "handling_greeting", {})
    
    response_content = get_greeting_response()
    
    logger.info("Greeting handled successfully")
    return {
        "messages": [AIMessage(content=response_content)],
        "next_agent": "FINISH",
    }


def help_handler_node(state: SupervisorState) -> dict:
    """
    Handle help/capability queries with platform information.
    
    This node responds to help requests without invoking any agents.
    Uses the pre-defined help response for consistency.
    """
    log_agent_action(logger, "help_handler", "handling_help_request", {})
    
    response_content = get_help_response()
    
    logger.info("Help request handled successfully")
    return {
        "messages": [AIMessage(content=response_content)],
        "next_agent": "FINISH",
    }


def ma_question_handler_node(state: SupervisorState) -> dict:
    """
    Handle M&A-related questions without specific company context.
    
    This node answers conceptual questions about M&A, due diligence,
    synergies, etc. without invoking the full agent chain.
    It also prompts the user to provide company names if they want
    to perform actual due diligence.
    """
    log_agent_action(logger, "ma_question_handler", "handling_ma_question", {})
    
    llm = get_llm(temperature=0.1)
    
    last_message = get_last_human_message(state.messages)
    
    prompt = f"""You are an M&A Due Diligence expert. Answer this question about M&A concepts.

Question: {last_message}

Provide a clear, educational answer about the M&A concept being asked.

At the end of your response, include this note:
---
💡 **Want to perform actual due diligence?**
To start a comprehensive due diligence analysis, please provide:
1. The acquiring company name
2. The target company name  
3. Whether this is a merger or acquisition

Example: "Analyze the merger between CompanyA and CompanyB"
"""
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        logger.info("M&A question handled successfully")
        return {
            "messages": [response],
            "next_agent": "FINISH",
        }
    except Exception as e:
        logger.error(f"Error handling M&A question: {e}")
        return {
            "messages": [AIMessage(content=f"I apologize, I encountered an error while processing your question. Please try again. Error: {str(e)}")],
            "next_agent": "FINISH",
        }


def informational_handler_node(state: SupervisorState) -> dict:
    """
    Handle general informational queries not related to M&A.
    
    This node handles queries that are either:
    - General information requests unrelated to M&A
    - Queries classified as UNKNOWN
    
    It politely redirects users to M&A-related topics.
    """
    log_agent_action(logger, "informational_handler", "handling_informational_query", {})
    
    llm = get_llm(temperature=0.1)
    
    last_message = get_last_human_message(state.messages)
    
    prompt = f"""You are an M&A Due Diligence Assistant. The user asked a question that may not be directly related to M&A.

User's Question: {last_message}

If this question is related to M&A or business topics, provide helpful information.
If this question is completely unrelated to M&A (like weather, jokes, etc.), politely acknowledge their question briefly and then redirect them to M&A topics.

Always end your response by mentioning that you specialize in:
- 📊 Financial due diligence
- ⚖️ Legal due diligence  
- 👥 HR due diligence
- 📈 Strategic analysis for mergers and acquisitions

And that to start an analysis, they should provide the names of the companies involved in the merger or acquisition.
"""
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        logger.info("Informational query handled successfully")
        return {
            "messages": [response],
            "next_agent": "FINISH",
        }
    except Exception as e:
        logger.error(f"Error handling informational query: {e}")
        return {
            "messages": [AIMessage(content="""I'm an M&A Due Diligence Assistant. I specialize in:

📊 **Financial Due Diligence** - Analyze financial statements and identify risks
⚖️ **Legal Due Diligence** - Review contracts, litigation, and compliance
👥 **HR Due Diligence** - Assess workforce and cultural fit
📈 **Strategic Analysis** - Evaluate synergies and deal recommendations

To start an analysis, please provide the names of the companies involved in your merger or acquisition.

Example: "Analyze the acquisition of TargetCorp by AcquirerInc"
""")],
            "next_agent": "FINISH",
        }


# =============================================================================
# END HANDLER NODES
# =============================================================================


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
    
    Note: Greetings and simple queries are now handled by the intent classifier
    and handler nodes before reaching this node. This node is only invoked for
    MA_DUE_DILIGENCE intents with company names.
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
    """
    Determine default routing based on current phase and chain activation.
    
    If chain_activated is False, we should not invoke any agents and should
    finish immediately. This is a safety check for edge cases.
    """
    
    # Safety check: If chain is not activated, finish immediately
    if not state.chain_activated:
        logger.warning("Chain not activated, returning FINISH from default routing")
        return {"next_agent": "FINISH", "reasoning": "Chain not activated - no company names provided"}
    
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
    
    Graph Structure (Updated with Intent Classification):
    
    START → intent_classifier → [route_after_intent]
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
                ▼                      ▼                      ▼
         greeting_handler      ma_question_handler     supervisor
         help_handler          informational_handler       │
                │                      │            [route_to_agent]
                │                      │                   │
                ▼                      ▼         ┌─────────┴─────────┐
               END                    END        │                   │
                                           ┌─────┴─────┐       human_review
                                           │  agents   │             │
                                           └─────┬─────┘             ▼
                                                 │                  END
                                                 ▼
                                            supervisor
                                                 │
                                                 ▼
                                                END
    
    The intent_classifier node runs first to:
    1. Classify user intent (GREETING, HELP, MA_QUESTION, MA_DUE_DILIGENCE, etc.)
    2. Extract company names for M&A requests
    3. Set chain_activated flag
    
    Only MA_DUE_DILIGENCE with company names routes to supervisor and triggers
    the full agent chain. All other intents are handled by specialized handlers.
    """
    
    workflow = StateGraph(SupervisorState)
    
    # =========================================================================
    # Add Intent Classification & Handler Nodes (Phase 5 & 6)
    # =========================================================================
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("greeting_handler", greeting_handler_node)
    workflow.add_node("help_handler", help_handler_node)
    workflow.add_node("ma_question_handler", ma_question_handler_node)
    workflow.add_node("informational_handler", informational_handler_node)
    
    # =========================================================================
    # Add Supervisor & Agent Nodes (existing)
    # =========================================================================
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("finance_agent", finance_agent_node)
    workflow.add_node("legal_agent", legal_agent_node)
    workflow.add_node("hr_agent", hr_agent_node)
    workflow.add_node("analyst_agent", analyst_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("human_review", human_review_node)
    
    # =========================================================================
    # Set Entry Point to Intent Classifier (NEW - Phase 6)
    # =========================================================================
    workflow.set_entry_point("intent_classifier")
    
    # =========================================================================
    # Route from Intent Classifier based on Intent (NEW - Phase 6)
    # =========================================================================
    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "supervisor": "supervisor",
            "greeting_handler": "greeting_handler",
            "help_handler": "help_handler",
            "ma_question_handler": "ma_question_handler",
            "informational_handler": "informational_handler",
        }
    )
    
    # =========================================================================
    # Handler Nodes go directly to END (NEW - Phase 6)
    # =========================================================================
    workflow.add_edge("greeting_handler", END)
    workflow.add_edge("help_handler", END)
    workflow.add_edge("ma_question_handler", END)
    workflow.add_edge("informational_handler", END)
    
    # =========================================================================
    # Existing Routing from Supervisor (unchanged)
    # =========================================================================
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
    
    # Human review goes to END
    workflow.add_edge("human_review", END)
    
    return workflow.compile()


# Export the compiled graph for langgraph.json
graph = build_supervisor_graph()
