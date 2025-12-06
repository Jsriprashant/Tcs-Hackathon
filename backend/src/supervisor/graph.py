"""
Supervisor Agent Graph - Main orchestrator for M&A Due Diligence.

ENHANCED v2.0:
- Intelligent intent classification with scope detection
- Dynamic analysis planning
- Risk aggregation with weighted scoring
- Master analyst for final recommendation
- Support for single-domain and full analyses

This module implements the main LangGraph workflow that coordinates
all specialized agents for comprehensive due diligence analysis.
"""

from typing import Literal, Any, List, Dict, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import time

from src.config.llm_config import get_llm
from src.supervisor.state import SupervisorState
from src.supervisor.prompts import (
    SUPERVISOR_SYSTEM_PROMPT, 
    ROUTING_PROMPT,
    MASTER_ANALYST_PROMPT,
    INTELLIGENT_ROUTING_PROMPT,
    DOMAIN_SUMMARY_PROMPT,
    RISK_AGGREGATION_PROMPT,
    GREETING_PROMPT,
    HELP_PROMPT,
    INFORMATIONAL_REDIRECT_PROMPT,
)
from src.common.logging_config import get_logger, log_agent_action
from src.common.guardrails import PIIFilter, InputValidator
from src.common.state import CompanyInfo

# Import intent classifier (Enhanced)
from src.common.intent_classifier import (
    classify_intent,
    classify_intent_enhanced,
    get_last_human_message,
    IntentType,
    IntentClassificationResult,
)

# Import enhanced models
from src.supervisor.models import (
    AnalysisScope,
    DealType,
    EnhancedIntentResult,
    AgentOutput,
    DomainRiskScore,
    AggregatedRisk,
    DealAnalysis,
    ReasoningStep,
    RiskLevel,
    Recommendation,
    DOMAIN_WEIGHTS,
    calculate_risk_level,
    get_recommendation_from_risk,
)

# Import planner
from src.supervisor.planner import (
    create_analysis_plan,
    get_next_agents,
    is_plan_complete,
    get_pending_agents,
)

# Import sub-agents
from src.finance_agent.graph import finance_agent
from src.legal_agent.graph import legal_agent
from src.hr_agent.graph import hr_agent
from src.analyst_agent.graph import analyst_agent
from src.rag_agent.graph import rag_agent

# Import parsers for agent I/O transformation
from src.supervisor.parsers import (
    create_legal_agent_input,
    parse_legal_agent_output,
    create_finance_agent_input,
    parse_finance_agent_output,
    create_hr_agent_input,
    parse_hr_agent_output,
    ConsolidatedResult,
)

logger = get_logger(__name__)

# Agent mapping
AGENTS = {
    "finance_agent": finance_agent,
    "legal_agent": legal_agent,
    "hr_agent": hr_agent,
    "analyst_agent": analyst_agent,
    "rag_agent": rag_agent,
}


# =============================================================================
# INTENT CLASSIFICATION NODE (Enhanced v2.0)
# This node runs FIRST to classify user intent and gate chain activation
# =============================================================================

def intent_classifier_node(state: SupervisorState) -> dict:
    """
    Enhanced intent classification with analysis scope detection.
    
    This node runs first, before any routing decision. It determines:
    1. Intent type (MA_DUE_DILIGENCE, MA_QUESTION, etc.)
    2. Analysis scope (FULL, FINANCIAL_ONLY, LEGAL_ONLY, etc.)
    3. Company names (acquirer and/or target)
    4. Required domains for analysis
    
    Args:
        state: Current supervisor state
        
    Returns:
        Dict with intent classification results and optional company info
    """
    log_agent_action(logger, "intent_classifier", "classifying_intent_enhanced", {})
    
    # Get last user message
    last_message = get_last_human_message(state.messages)
    
    if not last_message:
        logger.warning("No human message found for intent classification")
        return {
            "intent_classified": True,
            "intent_type": "UNKNOWN",
            "intent_confidence": 0.0,
            "chain_activated": False,
            "execution_phase": "complete",
        }
    
    # Use enhanced classification
    enhanced_result: EnhancedIntentResult = classify_intent_enhanced(last_message)
    
    log_agent_action(logger, "intent_classifier", "enhanced_classification_result", {
        "intent": enhanced_result.intent,
        "confidence": enhanced_result.confidence,
        "analysis_scope": enhanced_result.analysis_scope.value if enhanced_result.analysis_scope else None,
        "acquirer": enhanced_result.acquirer_company,
        "target": enhanced_result.target_company,
        "required_domains": enhanced_result.required_domains,
        "should_activate_chain": enhanced_result.should_activate_chain,
    })
    
    # Build state updates
    updates = {
        "intent_classified": True,
        "intent_type": enhanced_result.intent,
        "intent_confidence": enhanced_result.confidence,
        "chain_activated": enhanced_result.should_activate_chain,
        "enhanced_intent": enhanced_result,
        "analysis_scope": enhanced_result.analysis_scope,
        "required_domains": enhanced_result.required_domains,
        "execution_phase": "intent_classification",
    }
    
    # Extract and set companies if M&A due diligence with company names
    if enhanced_result.intent == "MA_DUE_DILIGENCE" and enhanced_result.should_activate_chain:
        if enhanced_result.acquirer_company:
            updates["acquirer"] = CompanyInfo(
                company_id=enhanced_result.acquirer_company.lower().replace(" ", "_"),
                company_name=enhanced_result.acquirer_company,
                industry="Unknown"  # Will be populated by RAG agent
            )
            logger.info(f"Set acquirer company: {enhanced_result.acquirer_company}")
        
        if enhanced_result.target_company:
            updates["target"] = CompanyInfo(
                company_id=enhanced_result.target_company.lower().replace(" ", "_"),
                company_name=enhanced_result.target_company,
                industry="Unknown"  # Will be populated by RAG agent
            )
            # Also set as active context for follow-up queries
            updates["active_company_context"] = enhanced_result.target_company
            logger.info(f"Set target company: {enhanced_result.target_company}")
        
        # Set deal type
        if enhanced_result.deal_type:
            updates["deal_type"] = enhanced_result.deal_type.value
        
        # Track domains for session context
        if enhanced_result.required_domains:
            updates["last_analyzed_domains"] = enhanced_result.required_domains
    
    logger.info(
        f"Enhanced intent classification complete: {enhanced_result.intent}, "
        f"scope={enhanced_result.analysis_scope.value if enhanced_result.analysis_scope else 'N/A'}, "
        f"chain_activated={enhanced_result.should_activate_chain}"
    )
    
    return updates


def route_after_intent(state: SupervisorState) -> str:
    """
    Route based on classified intent.
    
    This function is called after intent_classifier_node to determine
    the next node in the graph based on the classified intent.
    
    ENHANCED: Routes to analysis_planner for MA_DUE_DILIGENCE before supervisor.
    NEW: Routes to clarification_handler for DOMAIN_QUERY_NO_CONTEXT.
    
    Args:
        state: Current supervisor state with intent classification
        
    Returns:
        String name of the next node to execute
    """
    intent = state.intent_type
    
    log_agent_action(logger, "intent_router", "routing_decision", {
        "intent": intent,
        "chain_activated": state.chain_activated,
        "analysis_scope": state.analysis_scope.value if state.analysis_scope else None,
        "active_company_context": state.active_company_context,
    })
    
    # MA_DUE_DILIGENCE with companies -> go to analysis planner first
    if intent == "MA_DUE_DILIGENCE" and state.chain_activated:
        logger.info("Routing to analysis_planner for execution planning")
        return "analysis_planner"
    
    # DOMAIN_QUERY_NO_CONTEXT -> needs clarification (or use session context)
    elif intent == "DOMAIN_QUERY_NO_CONTEXT":
        logger.info("Routing to clarification_handler for company context")
        return "clarification_handler"
    
    # FOLLOW_UP -> check session context and route accordingly
    elif intent == "FOLLOW_UP":
        if state.active_company_context:
            logger.info(f"Follow-up with context: {state.active_company_context}, routing to clarification")
            return "clarification_handler"
        else:
            logger.info("Follow-up without context, routing to clarification")
            return "clarification_handler"
    
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
    Handle greeting queries with a friendly, LLM-generated response.
    
    This node responds to simple greetings without invoking any agents.
    Uses LLM to generate natural, contextual responses.
    """
    log_agent_action(logger, "greeting_handler", "handling_greeting", {})
    
    llm = get_llm(temperature=0.3)  # Slightly higher for natural greeting
    last_message = get_last_human_message(state.messages) or "Hello"
    
    prompt = GREETING_PROMPT.format(user_message=last_message)
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        logger.info("Greeting handled successfully with LLM")
        return {
            "messages": [response],
            "next_agent": "FINISH",
        }
    except Exception as e:
        logger.error(f"Error generating greeting: {e}")
        return {
            "messages": [AIMessage(content="Hello! I'm your M&A Due Diligence Assistant. How can I help you today?")],
            "next_agent": "FINISH",
        }


def help_handler_node(state: SupervisorState) -> dict:
    """
    Handle help/capability queries with LLM-generated platform information.
    
    This node responds to help requests without invoking any agents.
    Uses LLM to generate comprehensive, contextual help responses.
    """
    log_agent_action(logger, "help_handler", "handling_help_request", {})
    
    llm = get_llm(temperature=0.2)
    last_message = get_last_human_message(state.messages) or "What can you do?"
    
    prompt = HELP_PROMPT.format(user_message=last_message)
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        logger.info("Help request handled successfully with LLM")
        return {
            "messages": [response],
            "next_agent": "FINISH",
        }
    except Exception as e:
        logger.error(f"Error generating help response: {e}")
        return {
            "messages": [AIMessage(content="I can help with M&A due diligence including financial, legal, and HR analysis. Please provide a company name to get started.")],
            "next_agent": "FINISH",
        }


def ma_question_handler_node(state: SupervisorState) -> dict:
    """
    Handle M&A-related questions without specific company context.
    
    This node answers conceptual questions about M&A, due diligence,
    synergies, etc. without invoking the full agent chain.
    Uses LLM to provide educational, contextual responses.
    
    ENHANCED: Now checks if query actually contains a company name that was
    missed during classification. If so, redirects to proper analysis.
    """
    log_agent_action(logger, "ma_question_handler", "handling_ma_question", {
        "active_company_context": state.active_company_context,
    })
    
    last_message = get_last_human_message(state.messages) or ""
    
    # ENHANCED: Check if this query actually mentions a company name
    # that should trigger analysis instead of a generic answer
    from src.common.intent_classifier import extract_potential_company_names, has_domain_keywords
    potential_companies = extract_potential_company_names(last_message)
    has_domains, matched_domains = has_domain_keywords(last_message)
    
    # If we found a company name AND domain keywords, this should be analysis, not Q&A
    if potential_companies and matched_domains:
        company_name = potential_companies[0]
        logger.info(f"MA_QUESTION handler detected company '{company_name}' with domains {matched_domains} - redirecting to analysis")
        
        # Determine scope based on domains
        if len(matched_domains) == 1:
            domain = matched_domains[0]
            scope_map = {
                "finance": AnalysisScope.FINANCIAL_ONLY,
                "legal": AnalysisScope.LEGAL_ONLY,
                "hr": AnalysisScope.HR_ONLY,
                "compliance": AnalysisScope.COMPLIANCE_ONLY,
                "strategic": AnalysisScope.STRATEGIC_ONLY,
            }
            analysis_scope = scope_map.get(domain, AnalysisScope.FULL_DUE_DILIGENCE)
        else:
            analysis_scope = AnalysisScope.FULL_DUE_DILIGENCE
        
        # Create focused query
        focused_query = create_focused_query(last_message, matched_domains)
        
        return {
            "messages": [AIMessage(content=f"I'll analyze {company_name}'s {', '.join(matched_domains)} data for you...")],
            "chain_activated": True,
            "intent_type": "MA_DUE_DILIGENCE",
            "analysis_scope": analysis_scope,
            "required_domains": matched_domains,
            "focused_query": focused_query,
            "target": CompanyInfo(
                company_id=company_name.lower().replace(" ", "_"),
                company_name=company_name,
                industry="Unknown"
            ),
            "active_company_context": company_name,
            "next_agent": "analysis_planner",
        }
    
    # Check if there's active company context and domains - also redirect to analysis
    if state.active_company_context and matched_domains:
        logger.info(f"MA_QUESTION handler using context '{state.active_company_context}' with domains {matched_domains}")
        
        # Determine scope based on domains
        if len(matched_domains) == 1:
            domain = matched_domains[0]
            scope_map = {
                "finance": AnalysisScope.FINANCIAL_ONLY,
                "legal": AnalysisScope.LEGAL_ONLY,
                "hr": AnalysisScope.HR_ONLY,
                "compliance": AnalysisScope.COMPLIANCE_ONLY,
                "strategic": AnalysisScope.STRATEGIC_ONLY,
            }
            analysis_scope = scope_map.get(domain, AnalysisScope.FULL_DUE_DILIGENCE)
        else:
            analysis_scope = AnalysisScope.FULL_DUE_DILIGENCE
        
        focused_query = create_focused_query(last_message, matched_domains)
        
        return {
            "messages": [AIMessage(content=f"I'll analyze {state.active_company_context}'s {', '.join(matched_domains)} data...")],
            "chain_activated": True,
            "intent_type": "MA_DUE_DILIGENCE",
            "analysis_scope": analysis_scope,
            "required_domains": matched_domains,
            "focused_query": focused_query,
            "target": CompanyInfo(
                company_id=state.active_company_context.lower().replace(" ", "_"),
                company_name=state.active_company_context,
                industry="Unknown"
            ),
            "next_agent": "analysis_planner",
        }
    
    # Truly conceptual question - give educational answer
    llm = get_llm(temperature=0.2)
    
    prompt = f"""You are an expert M&A Due Diligence consultant. The user has asked a conceptual question about M&A.

User's Question: {last_message}

Provide a comprehensive, educational answer that:
1. Directly addresses their question with expert-level insight
2. Includes relevant examples or scenarios where helpful
3. Mentions any important considerations or nuances
4. If applicable, explains how this relates to due diligence

At the end, offer to help with specific company analysis if they'd like to proceed with actual due diligence.

Respond in a professional but conversational tone:"""
    
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
            "messages": [AIMessage(content=f"I apologize, I encountered an error while processing your question. Please try again.")],
            "next_agent": "FINISH",
        }


def informational_handler_node(state: SupervisorState) -> dict:
    """
    Handle general informational queries not related to M&A.
    
    This node handles queries that are either:
    - General information requests unrelated to M&A
    - Queries classified as UNKNOWN
    
    Uses LLM to provide helpful responses and redirect to M&A topics.
    """
    log_agent_action(logger, "informational_handler", "handling_informational_query", {})
    
    llm = get_llm(temperature=0.2)
    
    last_message = get_last_human_message(state.messages) or ""
    
    prompt = INFORMATIONAL_REDIRECT_PROMPT.format(user_message=last_message)
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        logger.info("Informational query handled successfully with LLM")
        return {
            "messages": [response],
            "next_agent": "FINISH",
        }
    except Exception as e:
        logger.error(f"Error handling informational query: {e}")
        return {
            "messages": [AIMessage(content="I specialize in M&A due diligence. Please provide a company name to analyze, or ask me about financial, legal, or HR aspects of mergers and acquisitions.")],
            "next_agent": "FINISH",
        }


def create_focused_query(original_query: str, domains: List[str]) -> str:
    """
    Create a focused query context for domain agents.
    
    This function extracts the specific aspect the user is asking about
    and creates a focused prompt that tells agents exactly what to analyze.
    
    Args:
        original_query: The user's original query
        domains: List of detected domains
        
    Returns:
        Focused query string for agents
    """
    query_lower = original_query.lower()
    
    # Extract specific focus areas from the query
    focus_areas = []
    
    # Finance-specific focus detection
    if "finance" in domains:
        if any(kw in query_lower for kw in ["revenue", "sales", "top line"]):
            focus_areas.append("revenue and sales performance")
        if any(kw in query_lower for kw in ["profit", "margin", "earnings", "ebitda"]):
            focus_areas.append("profitability and margins")
        if any(kw in query_lower for kw in ["cash", "liquidity", "working capital"]):
            focus_areas.append("cash flow and liquidity")
        if any(kw in query_lower for kw in ["debt", "leverage", "liabilities"]):
            focus_areas.append("debt and leverage")
        if any(kw in query_lower for kw in ["growth", "trend", "trajectory"]):
            focus_areas.append("growth trends")
        if any(kw in query_lower for kw in ["valuation", "multiple", "worth"]):
            focus_areas.append("valuation metrics")
    
    # Legal-specific focus detection
    if "legal" in domains:
        if any(kw in query_lower for kw in ["litigation", "lawsuit", "court", "sue"]):
            focus_areas.append("litigation and legal disputes")
        if any(kw in query_lower for kw in ["contract", "agreement"]):
            focus_areas.append("contracts and agreements")
        if any(kw in query_lower for kw in ["ip", "patent", "trademark", "intellectual"]):
            focus_areas.append("intellectual property")
        if any(kw in query_lower for kw in ["compliance", "regulatory", "regulation"]):
            focus_areas.append("regulatory compliance")
        if any(kw in query_lower for kw in ["risk", "liability"]):
            focus_areas.append("legal risks and liabilities")
    
    # HR-specific focus detection
    if "hr" in domains:
        if any(kw in query_lower for kw in ["employee", "headcount", "staff", "workforce"]):
            focus_areas.append("workforce and headcount")
        if any(kw in query_lower for kw in ["attrition", "turnover", "retention"]):
            focus_areas.append("employee retention and attrition")
        if any(kw in query_lower for kw in ["culture", "morale", "engagement"]):
            focus_areas.append("culture and employee engagement")
        if any(kw in query_lower for kw in ["compensation", "salary", "benefits", "pay"]):
            focus_areas.append("compensation and benefits")
        if any(kw in query_lower for kw in ["talent", "key person", "leadership"]):
            focus_areas.append("key personnel and talent")
    
    if focus_areas:
        return f"FOCUS ANALYSIS ON: {', '.join(focus_areas)}. Original query: {original_query}"
    else:
        return f"ANALYZE: {original_query}"


def clarification_handler_node(state: SupervisorState) -> dict:
    """
    Handle actionable domain queries that need company context clarification.
    
    This node is triggered when:
    - User asks a specific domain question (e.g., "how is the company performing on revenue?")
    - But no specific company name is provided
    - The query is actionable (not just conceptual)
    
    If there's an active company in session context, use it.
    Otherwise, ask for clarification.
    
    ENHANCED: Now properly sets analysis_scope based on detected domains
    to enable single-agent routing for focused queries.
    """
    log_agent_action(logger, "clarification_handler", "handling_clarification", {
        "active_company_context": state.active_company_context,
    })
    
    last_message = get_last_human_message(state.messages) or ""
    
    # Check if we have an active company context from previous conversation
    if state.active_company_context:
        # We have context! Route to analysis with the known company
        logger.info(f"Using active company context: {state.active_company_context}")
        
        # Detect which domains are being asked about
        from src.common.intent_classifier import has_domain_keywords, detect_required_domains
        has_domains, matched_domains = has_domain_keywords(last_message)
        
        # Determine analysis scope based on matched domains
        # This is KEY - single domain = single agent routing
        if matched_domains and len(matched_domains) == 1:
            domain = matched_domains[0]
            scope_map = {
                "finance": AnalysisScope.FINANCIAL_ONLY,
                "legal": AnalysisScope.LEGAL_ONLY,
                "hr": AnalysisScope.HR_ONLY,
                "compliance": AnalysisScope.COMPLIANCE_ONLY,
                "strategic": AnalysisScope.STRATEGIC_ONLY,
            }
            analysis_scope = scope_map.get(domain, AnalysisScope.FULL_DUE_DILIGENCE)
            logger.info(f"Single domain detected ({domain}), setting scope to {analysis_scope.value}")
        elif matched_domains and len(matched_domains) > 1:
            # Multiple domains but not all - still targeted
            analysis_scope = AnalysisScope.FULL_DUE_DILIGENCE
            logger.info(f"Multiple domains detected: {matched_domains}, using FULL scope with domain filter")
        else:
            # No specific domains detected - full analysis
            analysis_scope = AnalysisScope.FULL_DUE_DILIGENCE
            matched_domains = ["finance", "legal", "hr"]
            logger.info("No specific domain detected, defaulting to full analysis")
        
        # Create focused query context for the agent
        focused_query = create_focused_query(last_message, matched_domains)
        
        # Create a clarified message that includes the company
        domain_desc = ", ".join(matched_domains) if matched_domains else "all aspects"
        
        # Set up for analysis with context
        return {
            "messages": [AIMessage(content=f"I'll analyze {state.active_company_context}'s {domain_desc}. Let me pull the relevant data...")],
            "chain_activated": True,
            "intent_type": "MA_DUE_DILIGENCE",
            "analysis_scope": analysis_scope,
            "required_domains": matched_domains,
            "focused_query": focused_query,  # Pass focused context to agents
            "target": CompanyInfo(
                company_id=state.active_company_context.lower().replace(" ", "_"),
                company_name=state.active_company_context,
                industry="Unknown"
            ),
            "next_agent": "analysis_planner",
        }
    
    # No context - need to ask for company name
    llm = get_llm(temperature=0.3)
    
    # Detect what domain they're asking about to make the clarification specific
    from src.common.intent_classifier import has_domain_keywords
    has_domains, matched_domains = has_domain_keywords(last_message)
    
    domain_context = ""
    if matched_domains:
        domain_names = {
            "finance": "financial metrics and performance",
            "legal": "legal and compliance status",
            "hr": "HR policies and workforce",
            "compliance": "regulatory compliance",
            "strategic": "strategic positioning"
        }
        domain_desc = [domain_names.get(d, d) for d in matched_domains]
        domain_context = f"I can see you're interested in {', '.join(domain_desc)}. "
    
    prompt = f"""You are an M&A Due Diligence Assistant. The user asked an actionable question but didn't specify which company to analyze.

User's Query: "{last_message}"

{domain_context}Generate a helpful, concise clarification request that:
1. Acknowledges what they're asking about
2. Asks specifically which company they want to analyze
3. Optionally mentions what analysis you can provide once they name a company

Keep it friendly and under 3 sentences. Don't be overly formal."""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        logger.info("Clarification request generated successfully")
        return {
            "messages": [response],
            "next_agent": "FINISH",
        }
    except Exception as e:
        logger.error(f"Error generating clarification: {e}")
        
        # Fallback clarification
        fallback_msg = f"I'd be happy to help with that analysis! {domain_context}Could you please tell me which company you'd like me to analyze?"
        return {
            "messages": [AIMessage(content=fallback_msg)],
            "next_agent": "FINISH",
        }


def route_after_clarification(state: SupervisorState) -> str:
    """
    Route after clarification handler.
    
    If the clarification handler found active company context and set up
    for analysis, route to analysis_planner. Otherwise, END (asked for clarification).
    
    Args:
        state: Current supervisor state
        
    Returns:
        Next node: "analysis_planner" or END
    """
    # Check if chain was activated (meaning we found context)
    if state.chain_activated and state.target:
        logger.info(f"Clarification found context, routing to analysis_planner for {state.target.company_name}")
        return "analysis_planner"
    
    # Otherwise, we asked for clarification - end turn
    logger.info("Clarification requested, ending turn")
    return END


# =============================================================================
# END HANDLER NODES
# =============================================================================


# =============================================================================
# NEW NODES (v2.0): Analysis Planner, Risk Aggregator, Master Analyst
# =============================================================================

def analysis_planner_node(state: SupervisorState) -> dict:
    """
    Creates execution plan based on enhanced intent classification.
    
    This node determines:
    - Which agents to invoke based on analysis scope
    - Execution order (sequential, parallel, hybrid)
    - Output format requirements
    
    Args:
        state: Current supervisor state with enhanced intent
        
    Returns:
        Dict with analysis plan and initial agent routing
    """
    log_agent_action(logger, "analysis_planner", "creating_plan", {
        "analysis_scope": state.analysis_scope.value if state.analysis_scope else "FULL",
    })
    
    # Get enhanced intent or create default
    if state.enhanced_intent:
        intent_result = state.enhanced_intent
    else:
        # Fallback: create basic intent result
        intent_result = EnhancedIntentResult(
            intent="MA_DUE_DILIGENCE",
            confidence=0.8,
            acquirer_company=state.acquirer.company_name if state.acquirer else None,
            target_company=state.target.company_name if state.target else None,
            analysis_scope=state.analysis_scope or AnalysisScope.FULL_DUE_DILIGENCE,
            required_domains=state.required_domains or ["finance", "legal", "hr"],
            deal_type=DealType(state.deal_type) if state.deal_type else DealType.ACQUISITION,
            depth="standard",
            should_activate_chain=True,
            reasoning="Fallback intent creation"
        )
    
    # Create analysis plan
    plan = create_analysis_plan(intent_result)
    
    log_agent_action(logger, "analysis_planner", "plan_created", {
        "plan_id": plan.plan_id,
        "required_agents": plan.required_agents,
        "execution_mode": plan.execution_mode,
        "report_format": plan.report_format,
    })
    
    # Determine first agents to run (always starts with RAG)
    first_agents = get_next_agents(plan, [], [])
    
    return {
        "analysis_plan": plan,
        "agents_pending": get_pending_agents(plan, [], []),
        "agents_completed": [],
        "execution_phase": "document_retrieval",
        "current_phase": "document_retrieval",
        "next_agent": first_agents[0] if first_agents else "FINISH",
    }


def risk_aggregator_node(state: SupervisorState) -> dict:
    """
    Aggregates risk scores from all domain agents.
    
    Uses weighted average with deal-breaker detection.
    Calculates overall risk score and level.
    
    Args:
        state: Current supervisor state with agent outputs
        
    Returns:
        Dict with aggregated risk assessment
    """
    log_agent_action(logger, "risk_aggregator", "aggregating_risks", {
        "completed_agents": state.agents_completed,
    })
    
    domain_risk_scores = {}
    deal_breakers = []
    key_concerns = []
    positive_factors = []
    
    # Process each agent output to extract risk scores
    for agent_name in state.agents_completed:
        if agent_name == "rag_agent":
            continue  # RAG doesn't produce risk scores
        
        domain = agent_name.replace("_agent", "")
        
        # Try to get structured output
        if agent_name in state.agent_outputs:
            output = state.agent_outputs[agent_name]
            risk_score = output.risk_score
            risk_factors = [rf.name for rf in output.risk_factors] if output.risk_factors else []
            recommendations = output.recommendations
        else:
            # Fallback: estimate from legacy results
            risk_score = estimate_risk_from_legacy(state, agent_name)
            risk_factors = []
            recommendations = []
        
        # Create domain risk score
        weight = DOMAIN_WEIGHTS.get(domain, 0.2)
        domain_risk_scores[domain] = DomainRiskScore(
            domain=domain,
            score=risk_score,
            weight=weight,
            contributing_factors=risk_factors,
            mitigations=recommendations
        )
        
        # Check for high-risk items
        if risk_score > 0.7:
            key_concerns.append(f"{domain.title()}: High risk score ({risk_score:.2f})")
        elif risk_score < 0.3:
            positive_factors.append(f"{domain.title()}: Low risk ({risk_score:.2f})")
    
    # Calculate aggregated risk
    if domain_risk_scores:
        aggregated_risk = AggregatedRisk.from_domain_scores(domain_risk_scores)
        
        # Check for deal-breakers in outputs
        deal_breakers = identify_deal_breakers_from_outputs(state)
        if deal_breakers:
            # Apply deal-breaker penalty
            aggregated_risk.overall_score = min(aggregated_risk.overall_score + 0.25, 1.0)
            aggregated_risk.deal_breakers = deal_breakers
            aggregated_risk.deal_breaker_penalty_applied = True
            aggregated_risk.risk_level = calculate_risk_level(aggregated_risk.overall_score)
        
        aggregated_risk.key_concerns = key_concerns
        aggregated_risk.positive_factors = positive_factors
    else:
        # No domain scores available
        aggregated_risk = AggregatedRisk(
            overall_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.5
        )
    
    log_agent_action(logger, "risk_aggregator", "aggregation_complete", {
        "overall_score": aggregated_risk.overall_score,
        "risk_level": aggregated_risk.risk_level.value,
        "deal_breakers_count": len(deal_breakers),
    })
    
    # Generate consolidated result for frontend
    consolidated_result = None
    if state.target:
        finance_output = state.agent_outputs.get("finance_agent")
        legal_output = state.agent_outputs.get("legal_agent")
        hr_output = state.agent_outputs.get("hr_agent")
        
        consolidated = ConsolidatedResult(
            company_id=state.target.company_id,
            company_name=state.target.company_name,
            finance_output=finance_output,
            legal_output=legal_output,
            hr_output=hr_output,
        )
        consolidated_result = consolidated.to_dict()
        
        log_agent_action(logger, "risk_aggregator", "consolidated_result_generated", {
            "overall_health_score": consolidated_result["overall"]["overall_health_score"],
            "recommendation": consolidated_result["overall"]["recommendation"],
        })
    
    return {
        "domain_risk_scores": domain_risk_scores,
        "aggregated_risk": aggregated_risk,
        "overall_risk_score": aggregated_risk.overall_score,
        "overall_risk_level": aggregated_risk.risk_level,
        "consolidated_result": consolidated_result,
        "execution_phase": "recommendation_generation",
    }


def master_analyst_node(state: SupervisorState) -> dict:
    """
    Master analyst that synthesizes all findings into final recommendation.
    
    This node:
    1. Collects all agent outputs
    2. Uses aggregated risk scores
    3. Generates step-by-step reasoning
    4. Produces final GO/NO-GO/CONDITIONAL recommendation
    5. Formats comprehensive M&A report
    
    Args:
        state: Current supervisor state with all analyses
        
    Returns:
        Dict with deal analysis and final response
    """
    log_agent_action(logger, "master_analyst", "generating_recommendation", {
        "overall_risk": state.overall_risk_score,
        "completed_agents": state.agents_completed,
    })
    
    llm = get_llm(temperature=0.1)
    
    # Format agent outputs for the prompt
    agent_outputs_text = format_agent_outputs_for_prompt(state)
    
    # Build master analyst prompt
    prompt = MASTER_ANALYST_PROMPT.format(
        acquirer=state.acquirer.company_name if state.acquirer else "Unknown Acquirer",
        target=state.target.company_name if state.target else "Unknown Target",
        deal_type=state.deal_type or "acquisition",
        agent_outputs=agent_outputs_text,
    )
    
    try:
        messages = [SystemMessage(content=prompt)]
        response = llm.invoke(messages)
        
        # Determine recommendation based on aggregated risk
        aggregated_risk = state.aggregated_risk
        if aggregated_risk:
            has_deal_breakers = len(aggregated_risk.deal_breakers) > 0
            recommendation = get_recommendation_from_risk(
                aggregated_risk.overall_score, 
                has_deal_breakers
            )
        else:
            recommendation = Recommendation.CONDITIONAL
        
        # Create deal analysis record
        deal_analysis = DealAnalysis(
            recommendation=recommendation,
            recommendation_confidence=0.8,
            executive_summary=extract_executive_summary(response.content),
            reasoning_chain=extract_reasoning_chain(response.content),
            aggregated_risk=aggregated_risk,
            agents_consulted=state.agents_completed,
        )
        
        log_agent_action(logger, "master_analyst", "recommendation_generated", {
            "recommendation": recommendation.value,
            "overall_risk": aggregated_risk.overall_score if aggregated_risk else "N/A",
        })
        
        return {
            "messages": [response],
            "deal_analysis": deal_analysis,
            "deal_recommendation": recommendation.value,
            "recommendation_rationale": response.content[:500],
            "execution_phase": "complete",
            "current_phase": "complete",
            "next_agent": "FINISH",
        }
        
    except Exception as e:
        logger.error(f"Master analyst error: {e}")
        return {
            "messages": [AIMessage(content=f"Error generating final analysis: {str(e)}")],
            "next_agent": "FINISH",
            "errors": state.errors + [str(e)],
        }


def domain_summarizer_node(state: SupervisorState) -> dict:
    """
    Summarizes single-domain analysis for targeted queries.
    
    Used when user requests only financial, legal, or HR analysis
    without full due diligence.
    
    Args:
        state: Current supervisor state
        
    Returns:
        Dict with domain summary response
    """
    log_agent_action(logger, "domain_summarizer", "summarizing_domain", {
        "analysis_scope": state.analysis_scope.value if state.analysis_scope else "N/A",
    })
    
    llm = get_llm(temperature=0.1)
    
    # Determine which domain was analyzed
    domain = None
    if state.analysis_scope == AnalysisScope.FINANCIAL_ONLY:
        domain = "Financial"
    elif state.analysis_scope == AnalysisScope.LEGAL_ONLY:
        domain = "Legal"
    elif state.analysis_scope == AnalysisScope.HR_ONLY:
        domain = "HR"
    elif state.analysis_scope == AnalysisScope.COMPLIANCE_ONLY:
        domain = "Compliance"
    else:
        domain = "Analysis"
    
    # Get the analysis results
    analysis_results = get_domain_analysis_text(state, domain.lower())
    
    prompt = DOMAIN_SUMMARY_PROMPT.format(
        domain=domain.lower(),
        domain_title=domain,
        company_name=state.target.company_name if state.target else "Target Company",
        analysis_results=analysis_results,
    )
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        
        return {
            "messages": [response],
            "execution_phase": "complete",
            "current_phase": "complete",
            "next_agent": "FINISH",
        }
    except Exception as e:
        logger.error(f"Domain summarizer error: {e}")
        return {
            "messages": [AIMessage(content=f"Error summarizing analysis: {str(e)}")],
            "next_agent": "FINISH",
        }


# =============================================================================
# HELPER FUNCTIONS FOR NEW NODES
# =============================================================================

def estimate_risk_from_legacy(state: SupervisorState, agent_name: str) -> float:
    """Estimate risk score from legacy result fields."""
    if agent_name == "finance_agent" and state.finance_result:
        return state.finance_result.risk_score.score if state.finance_result.risk_score else 0.5
    elif agent_name == "legal_agent" and state.legal_result:
        return state.legal_result.risk_score.score if state.legal_result.risk_score else 0.5
    elif agent_name == "hr_agent" and state.hr_result:
        return state.hr_result.risk_score.score if state.hr_result.risk_score else 0.5
    return 0.5


def identify_deal_breakers_from_outputs(state: SupervisorState) -> List[str]:
    """Identify deal-breakers from agent outputs."""
    deal_breakers = []
    
    for agent_name, output in state.agent_outputs.items():
        if hasattr(output, 'red_flags') and output.red_flags:
            for flag in output.red_flags:
                if "deal breaker" in flag.lower() or "critical" in flag.lower():
                    deal_breakers.append(flag)
        
        if hasattr(output, 'risk_factors'):
            for rf in output.risk_factors:
                if rf.is_deal_breaker:
                    deal_breakers.append(rf.name)
    
    return deal_breakers


def format_agent_outputs_for_prompt(state: SupervisorState) -> str:
    """Format agent outputs for master analyst prompt."""
    sections = []
    
    # Finance
    if "finance_agent" in state.agents_completed:
        if "finance_agent" in state.agent_outputs:
            output = state.agent_outputs["finance_agent"]
            sections.append(f"### Financial Analysis\n{output.summary}\nRisk Score: {output.risk_score}")
        elif state.finance_result:
            sections.append(f"### Financial Analysis\n{state.finance_result.summary}")
        else:
            sections.append("### Financial Analysis\nAnalysis completed.")
    
    # Legal
    if "legal_agent" in state.agents_completed:
        if "legal_agent" in state.agent_outputs:
            output = state.agent_outputs["legal_agent"]
            sections.append(f"### Legal Analysis\n{output.summary}\nRisk Score: {output.risk_score}")
        elif state.legal_result:
            sections.append(f"### Legal Analysis\n{state.legal_result.summary}")
        else:
            sections.append("### Legal Analysis\nAnalysis completed.")
    
    # HR
    if "hr_agent" in state.agents_completed:
        if "hr_agent" in state.agent_outputs:
            output = state.agent_outputs["hr_agent"]
            sections.append(f"### HR Analysis\n{output.summary}\nRisk Score: {output.risk_score}")
        elif state.hr_result:
            sections.append(f"### HR Analysis\n{state.hr_result.summary}")
        else:
            sections.append("### HR Analysis\nAnalysis completed.")
    
    # Include aggregated risk if available
    if state.aggregated_risk:
        sections.append(f"""
### Aggregated Risk Assessment
- Overall Risk Score: {state.aggregated_risk.overall_score:.2f}
- Risk Level: {state.aggregated_risk.risk_level.value.upper()}
- Deal Breakers: {', '.join(state.aggregated_risk.deal_breakers) if state.aggregated_risk.deal_breakers else 'None identified'}
- Key Concerns: {', '.join(state.aggregated_risk.key_concerns) if state.aggregated_risk.key_concerns else 'None'}
""")
    
    return "\n\n".join(sections) if sections else "No agent outputs available."


def get_domain_analysis_text(state: SupervisorState, domain: str) -> str:
    """Get analysis text for a specific domain."""
    agent_name = f"{domain}_agent"
    
    if agent_name in state.agent_outputs:
        output = state.agent_outputs[agent_name]
        return f"""
Summary: {output.summary}
Risk Score: {output.risk_score}
Key Findings: {', '.join(output.key_findings) if output.key_findings else 'N/A'}
Red Flags: {', '.join(output.red_flags) if output.red_flags else 'None'}
Recommendations: {', '.join(output.recommendations) if output.recommendations else 'N/A'}
"""
    
    # Fallback to legacy
    if domain == "financial" and state.finance_result:
        return state.finance_result.summary
    elif domain == "legal" and state.legal_result:
        return state.legal_result.summary
    elif domain == "hr" and state.hr_result:
        return state.hr_result.summary
    
    return "Analysis results not available."


def extract_executive_summary(content: str) -> str:
    """Extract executive summary from master analyst response."""
    # Try to find executive summary section
    if "EXECUTIVE SUMMARY" in content.upper():
        lines = content.split('\n')
        summary_lines = []
        in_summary = False
        for line in lines:
            if "EXECUTIVE SUMMARY" in line.upper():
                in_summary = True
                continue
            if in_summary:
                if line.startswith('###') or line.startswith('## '):
                    break
                summary_lines.append(line)
        return '\n'.join(summary_lines).strip()[:500]
    
    # Fallback: first paragraph
    paragraphs = content.split('\n\n')
    return paragraphs[0][:500] if paragraphs else content[:500]


def extract_reasoning_chain(content: str) -> List[ReasoningStep]:
    """Extract reasoning chain from master analyst response."""
    steps = []
    # Simple extraction - look for numbered items with arrows
    lines = content.split('\n')
    step_num = 0
    
    for line in lines:
        if '→' in line and (line.strip().startswith(str(step_num + 1)) or line.strip()[0].isdigit()):
            step_num += 1
            parts = line.split('→')
            if len(parts) >= 2:
                steps.append(ReasoningStep(
                    step_number=step_num,
                    analysis=parts[0].strip().lstrip('0123456789. '),
                    finding=parts[1].strip() if len(parts) > 1 else "",
                    implication=parts[2].strip() if len(parts) > 2 else "",
                    confidence=0.8
                ))
    
    return steps


# =============================================================================
# ROUTING FUNCTIONS FOR NEW NODES
# =============================================================================

def route_after_planning(state: SupervisorState) -> str:
    """Route after analysis planning to first agent."""
    if state.analysis_plan:
        next_agents = get_next_agents(state.analysis_plan, [], [])
        if next_agents:
            return next_agents[0]
    return "rag_agent"


def route_after_domain_agents(state: SupervisorState) -> str:
    """
    Route after domain agents complete.
    
    Checks if all required domain agents are done, then routes to:
    - risk_aggregator for full analysis
    - domain_summarizer for single-domain analysis
    """
    plan = state.analysis_plan
    
    if not plan:
        return "supervisor"  # Fallback to original flow
    
    # Check if plan is complete
    if is_plan_complete(plan, state.agents_completed, state.agents_failed):
        # Determine next step based on scope
        if plan.analysis_scope == AnalysisScope.FULL_DUE_DILIGENCE:
            return "risk_aggregator"
        elif plan.analysis_scope in [
            AnalysisScope.FINANCIAL_ONLY,
            AnalysisScope.LEGAL_ONLY,
            AnalysisScope.HR_ONLY,
            AnalysisScope.COMPLIANCE_ONLY,
        ]:
            return "domain_summarizer"
        else:
            return "risk_aggregator"
    
    # More agents to run
    next_agents = get_next_agents(plan, state.agents_completed, state.agents_failed)
    if next_agents:
        return next_agents[0]
    
    return "risk_aggregator"


def route_after_risk_aggregation(state: SupervisorState) -> str:
    """Route after risk aggregation to master analyst."""
    return "master_analyst"


# =============================================================================
# END NEW NODES
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
    """
    Invoke the finance agent with proper input/output transformation.
    
    Uses parsers to:
    1. Transform supervisor state → finance agent input format
    2. Parse finance agent output → AgentOutput model
    """
    log_agent_action(logger, "supervisor", "invoking_finance_agent", {
        "target": state.target.company_name if state.target else None,
    })
    
    try:
        # Create properly formatted input using parser
        agent_input = create_finance_agent_input(state)
        
        # Invoke finance agent
        result = finance_agent.invoke(agent_input)
        
        # Parse output into structured AgentOutput
        agent_output = parse_finance_agent_output(result)
        
        # Update agent_outputs dict
        new_agent_outputs = dict(state.agent_outputs)
        new_agent_outputs["finance_agent"] = agent_output
        
        log_agent_action(logger, "supervisor", "finance_agent_complete", {
            "risk_score": agent_output.risk_score,
            "risk_level": agent_output.risk_level.value,
        })
        
        # Update both legacy and new tracking
        new_agents_completed = list(state.agents_completed) + ["finance_agent"]
        
        return {
            "messages": result.get("messages", []),
            "agents_invoked": state.agents_invoked + ["finance_agent"],
            "agents_completed": new_agents_completed,
            "agent_outputs": new_agent_outputs,
            "current_phase": "legal_analysis",
            "execution_phase": "domain_analysis",
        }
        
    except ValueError as e:
        logger.error(f"Finance agent input error: {e}")
        return {
            "messages": [AIMessage(content=f"Finance analysis skipped: {str(e)}")],
            "agents_invoked": state.agents_invoked + ["finance_agent"],
            "agents_completed": list(state.agents_completed) + ["finance_agent"],
            "current_phase": "legal_analysis",
            "execution_phase": "domain_analysis",
        }
    except Exception as e:
        logger.error(f"Finance agent error: {e}")
        return {
            "messages": [AIMessage(content=f"Finance analysis encountered an error: {str(e)}")],
            "agents_invoked": state.agents_invoked + ["finance_agent"],
            "agents_completed": list(state.agents_completed) + ["finance_agent"],
            "current_phase": "legal_analysis",
            "execution_phase": "domain_analysis",
        }


def legal_agent_node(state: SupervisorState) -> dict:
    """
    Invoke the legal agent with proper input/output transformation.
    
    Uses parsers to:
    1. Transform supervisor state → legal agent input format (company_id)
    2. Parse legal agent output → AgentOutput model
    """
    log_agent_action(logger, "supervisor", "invoking_legal_agent", {
        "target": state.target.company_name if state.target else None,
    })
    
    try:
        # Create properly formatted input using parser
        agent_input = create_legal_agent_input(state)
        
        # Invoke legal agent
        result = legal_agent.invoke(agent_input)
        
        # Parse output into structured AgentOutput
        agent_output = parse_legal_agent_output(result)
        
        # Update agent_outputs dict
        new_agent_outputs = dict(state.agent_outputs)
        new_agent_outputs["legal_agent"] = agent_output
        
        log_agent_action(logger, "supervisor", "legal_agent_complete", {
            "risk_score": agent_output.risk_score,
            "risk_level": agent_output.risk_level.value,
            "findings_count": len(agent_output.findings),
        })
        
        new_agents_completed = list(state.agents_completed) + ["legal_agent"]
        
        return {
            "messages": result.get("messages", []),
            "agents_invoked": state.agents_invoked + ["legal_agent"],
            "agents_completed": new_agents_completed,
            "agent_outputs": new_agent_outputs,
            "current_phase": "hr_analysis",
            "execution_phase": "domain_analysis",
        }
        
    except ValueError as e:
        logger.error(f"Legal agent input error: {e}")
        return {
            "messages": [AIMessage(content=f"Legal analysis skipped: {str(e)}")],
            "agents_invoked": state.agents_invoked + ["legal_agent"],
            "agents_completed": list(state.agents_completed) + ["legal_agent"],
            "current_phase": "hr_analysis",
            "execution_phase": "domain_analysis",
        }
    except Exception as e:
        logger.error(f"Legal agent error: {e}")
        return {
            "messages": [AIMessage(content=f"Legal analysis encountered an error: {str(e)}")],
            "agents_invoked": state.agents_invoked + ["legal_agent"],
            "agents_completed": list(state.agents_completed) + ["legal_agent"],
            "current_phase": "hr_analysis",
            "execution_phase": "domain_analysis",
        }


def hr_agent_node(state: SupervisorState) -> dict:
    """
    Invoke the HR agent with proper input/output transformation.
    
    Uses parsers to:
    1. Transform supervisor state → HR agent input format (prompt with company)
    2. Parse HR agent output → AgentOutput model with compatibility score
    """
    log_agent_action(logger, "supervisor", "invoking_hr_agent", {
        "target": state.target.company_name if state.target else None,
    })
    
    try:
        # Create properly formatted input using parser
        agent_input = create_hr_agent_input(state)
        
        # Invoke HR agent
        result = hr_agent.invoke(agent_input)
        
        # Parse output into structured AgentOutput
        agent_output = parse_hr_agent_output(result)
        
        # Update agent_outputs dict
        new_agent_outputs = dict(state.agent_outputs)
        new_agent_outputs["hr_agent"] = agent_output
        
        log_agent_action(logger, "supervisor", "hr_agent_complete", {
            "risk_score": agent_output.risk_score,
            "risk_level": agent_output.risk_level.value,
            "findings_count": len(agent_output.findings),
        })
        
        new_agents_completed = list(state.agents_completed) + ["hr_agent"]
        
        return {
            "messages": result.get("messages", []),
            "agents_invoked": state.agents_invoked + ["hr_agent"],
            "agents_completed": new_agents_completed,
            "agent_outputs": new_agent_outputs,
            "current_phase": "strategic_analysis",
            "execution_phase": "domain_analysis",
        }
        
    except ValueError as e:
        logger.error(f"HR agent input error: {e}")
        return {
            "messages": [AIMessage(content=f"HR analysis skipped: {str(e)}")],
            "agents_invoked": state.agents_invoked + ["hr_agent"],
            "agents_completed": list(state.agents_completed) + ["hr_agent"],
            "current_phase": "strategic_analysis",
            "execution_phase": "domain_analysis",
        }
    except Exception as e:
        logger.error(f"HR agent error: {e}")
        return {
            "messages": [AIMessage(content=f"HR analysis encountered an error: {str(e)}")],
            "agents_invoked": state.agents_invoked + ["hr_agent"],
            "agents_completed": list(state.agents_completed) + ["hr_agent"],
            "current_phase": "strategic_analysis",
            "execution_phase": "domain_analysis",
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
    
    new_agents_completed = list(state.agents_completed) + ["analyst_agent"]
    
    return {
        "messages": result.get("messages", []),
        "agents_invoked": state.agents_invoked + ["analyst_agent"],
        "agents_completed": new_agents_completed,
        "current_phase": "complete",
        "execution_phase": "recommendation_generation",
    }


def rag_agent_node(state: SupervisorState) -> dict:
    """Invoke the RAG agent for document retrieval."""
    log_agent_action(logger, "supervisor", "invoking_rag_agent", {})
    
    # Convert CompanyInfo to dict for RAG agent compatibility
    target_dict = state.target.model_dump() if state.target else None
    
    result = rag_agent.invoke({
        "messages": state.messages,
        "target_company": target_dict,
    })
    
    new_agents_completed = list(state.agents_completed) + ["rag_agent"]
    
    return {
        "messages": result.get("messages", []),
        "agents_invoked": state.agents_invoked + ["rag_agent"],
        "agents_completed": new_agents_completed,
        "current_phase": "financial_analysis",
        "execution_phase": "domain_analysis",
    }


def human_review_node(state: SupervisorState) -> dict:
    """
    Node for human-in-the-loop review.
    Uses LLM to generate contextual review request based on findings.
    """
    log_agent_action(logger, "supervisor", "requesting_human_review", {
        "reason": state.human_review_reason or "High risk or critical decision required"
    })
    
    llm = get_llm(temperature=0.1)
    
    # Gather context for the review request
    review_reason = state.human_review_reason or "High risk factors identified"
    risk_level = state.overall_risk_level or "UNKNOWN"
    
    # Get recent findings
    recent_findings = []
    if state.agent_outputs:
        for output in state.agent_outputs[-3:]:  # Last 3 outputs
            if hasattr(output, 'key_findings'):
                recent_findings.extend(output.key_findings[:2])
    
    prompt = f"""You are an M&A Due Diligence system requesting human oversight.

Context:
- Review Reason: {review_reason}
- Current Risk Level: {risk_level}
- Key Findings: {recent_findings[:5] if recent_findings else 'Analysis in progress'}

Generate a professional, clear message requesting human review that:
1. Explains why human oversight is needed
2. Summarizes the key concerns requiring attention
3. Presents clear options for the reviewer (approve, request more analysis, modify thresholds, reject)
4. Maintains a professional tone suitable for M&A transactions

Format the response with clear headers and bullet points."""
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        return {
            "pending_human_review": True,
            "messages": [response],
        }
    except Exception as e:
        logger.error(f"Error generating human review request: {e}")
        return {
            "pending_human_review": True,
            "messages": [AIMessage(content=f"Human review required: {review_reason}. Please review the analysis and provide guidance.")],
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
    Build the enhanced supervisor graph that orchestrates all agents.
    
    ENHANCED Graph Structure (v2.0):
    
    START → intent_classifier → [route_after_intent]
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
    greeting_handler            ma_question_handler           analysis_planner
    help_handler                informational_handler               │
        │                              │                              ▼
        ▼                              ▼                      [domain agents]
       END                            END                    finance, legal, hr
                                                           (each calls RAG internally)
                                                                      │
                                                                      ▼
                                                              risk_aggregator
                                                                      │
                                                                      ▼
                                                              master_analyst
                                                                      │
                                                                      ▼
                                                                     END
    
    For single-domain queries:
    analysis_planner → [single domain agent] → domain_summarizer → END
    
    NOTE: Supervisor does NOT orchestrate RAG - each domain agent handles its own
    RAG calls internally for data retrieval. This simplifies the flow and allows
    agents to be more autonomous.
    
    The intent_classifier node runs first to:
    1. Classify user intent (GREETING, HELP, MA_QUESTION, MA_DUE_DILIGENCE, etc.)
    2. Detect analysis scope (FULL, FINANCIAL_ONLY, LEGAL_ONLY, etc.)
    3. Extract company names for M&A requests
    4. Set chain_activated flag
    """
    
    workflow = StateGraph(SupervisorState)
    
    # =========================================================================
    # PHASE 1: Intent Classification & Handler Nodes
    # =========================================================================
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("greeting_handler", greeting_handler_node)
    workflow.add_node("help_handler", help_handler_node)
    workflow.add_node("ma_question_handler", ma_question_handler_node)
    workflow.add_node("informational_handler", informational_handler_node)
    workflow.add_node("clarification_handler", clarification_handler_node)  # NEW: For queries needing context
    
    # =========================================================================
    # PHASE 2: Analysis Planning (NEW v2.0)
    # =========================================================================
    workflow.add_node("analysis_planner", analysis_planner_node)
    
    # =========================================================================
    # PHASE 3: Domain Agents (each handles its own RAG calls internally)
    # =========================================================================
    workflow.add_node("finance_agent", finance_agent_node)
    workflow.add_node("legal_agent", legal_agent_node)
    workflow.add_node("hr_agent", hr_agent_node)
    workflow.add_node("analyst_agent", analyst_agent_node)
    
    # Keep RAG node for legacy supervisor routing only
    workflow.add_node("rag_agent", rag_agent_node)
    
    # =========================================================================
    # PHASE 4: Risk Aggregation & Master Analysis (NEW v2.0)
    # =========================================================================
    workflow.add_node("risk_aggregator", risk_aggregator_node)
    workflow.add_node("master_analyst", master_analyst_node)
    workflow.add_node("domain_summarizer", domain_summarizer_node)
    
    # =========================================================================
    # PHASE 5: Legacy Supervisor & Human Review (backward compatibility)
    # =========================================================================
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("human_review", human_review_node)
    
    # =========================================================================
    # Set Entry Point to Intent Classifier
    # =========================================================================
    workflow.set_entry_point("intent_classifier")
    
    # =========================================================================
    # Route from Intent Classifier based on Intent
    # =========================================================================
    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "analysis_planner": "analysis_planner",  # NEW: goes to planner first
            "clarification_handler": "clarification_handler",  # NEW: queries needing context
            "supervisor": "supervisor",  # Legacy fallback
            "greeting_handler": "greeting_handler",
            "help_handler": "help_handler",
            "ma_question_handler": "ma_question_handler",
            "informational_handler": "informational_handler",
        }
    )
    
    # =========================================================================
    # Handler Nodes go directly to END
    # =========================================================================
    workflow.add_edge("greeting_handler", END)
    workflow.add_edge("help_handler", END)
    workflow.add_edge("ma_question_handler", END)
    workflow.add_edge("informational_handler", END)
    
    # =========================================================================
    # Clarification Handler routes to END or analysis_planner (if context found)
    # =========================================================================
    workflow.add_conditional_edges(
        "clarification_handler",
        route_after_clarification,
        {
            "analysis_planner": "analysis_planner",  # If context was found
            END: END,  # If asking for clarification
        }
    )
    
    # =========================================================================
    # Analysis Planner routes directly to first domain agent
    # =========================================================================
    workflow.add_conditional_edges(
        "analysis_planner",
        route_after_planning,
        {
            "finance_agent": "finance_agent",
            "legal_agent": "legal_agent",
            "hr_agent": "hr_agent",
            "analyst_agent": "analyst_agent",
            "supervisor": "supervisor",  # Fallback
        }
    )
    
    # =========================================================================
    # Domain Agents route to next agent or aggregation
    # =========================================================================
    workflow.add_conditional_edges(
        "finance_agent",
        route_after_domain_agent,
        {
            "legal_agent": "legal_agent",
            "hr_agent": "hr_agent",
            "risk_aggregator": "risk_aggregator",
            "domain_summarizer": "domain_summarizer",
            "supervisor": "supervisor",
        }
    )
    
    workflow.add_conditional_edges(
        "legal_agent",
        route_after_domain_agent,
        {
            "finance_agent": "finance_agent",
            "hr_agent": "hr_agent",
            "risk_aggregator": "risk_aggregator",
            "domain_summarizer": "domain_summarizer",
            "supervisor": "supervisor",
        }
    )
    
    workflow.add_conditional_edges(
        "hr_agent",
        route_after_domain_agent,
        {
            "finance_agent": "finance_agent",
            "legal_agent": "legal_agent",
            "risk_aggregator": "risk_aggregator",
            "domain_summarizer": "domain_summarizer",
            "supervisor": "supervisor",
        }
    )
    
    # =========================================================================
    # Risk Aggregator routes to Master Analyst
    # =========================================================================
    workflow.add_edge("risk_aggregator", "master_analyst")
    
    # =========================================================================
    # Master Analyst and Domain Summarizer route to END
    # =========================================================================
    workflow.add_edge("master_analyst", END)
    workflow.add_edge("domain_summarizer", END)
    
    # =========================================================================
    # Legacy Supervisor routing (backward compatibility)
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
            "risk_aggregator": "risk_aggregator",
            "master_analyst": "master_analyst",
            "human_review": "human_review",
            "supervisor": "supervisor",
            "end": END,
        }
    )
    
    # Analyst agent routes to supervisor (legacy) or risk_aggregator (new)
    workflow.add_conditional_edges(
        "analyst_agent",
        lambda s: "risk_aggregator" if s.analysis_plan else "supervisor",
        {
            "risk_aggregator": "risk_aggregator",
            "supervisor": "supervisor",
        }
    )
    
    # RAG agent routes back to supervisor (legacy mode only)
    workflow.add_edge("rag_agent", "supervisor")
    
    # Human review goes to END
    workflow.add_edge("human_review", END)
    
    return workflow.compile()


# =========================================================================
# NEW ROUTING FUNCTIONS
# =========================================================================

def route_after_planning(state: SupervisorState) -> str:
    """Route after analysis planning to first domain agent."""
    plan = state.analysis_plan
    
    if not plan:
        return "supervisor"
    
    # Get first agents from the plan
    next_agents = get_next_agents(plan, state.agents_completed, state.agents_failed)
    
    if next_agents:
        # Return the first domain agent
        for agent in next_agents:
            if agent in ["finance_agent", "legal_agent", "hr_agent", "analyst_agent"]:
                return agent
    
    return "supervisor"


def route_after_domain_agent(state: SupervisorState) -> str:
    """Route after a domain agent completes."""
    plan = state.analysis_plan
    
    if not plan:
        return "supervisor"
    
    # Check what's next in the plan
    next_agents = get_next_agents(plan, state.agents_completed, state.agents_failed)
    
    if next_agents:
        # Return next domain agent if available
        for agent in next_agents:
            if agent in ["finance_agent", "legal_agent", "hr_agent"]:
                return agent
    
    # All domain agents done - check scope
    if plan.analysis_scope in [
        AnalysisScope.FINANCIAL_ONLY,
        AnalysisScope.LEGAL_ONLY,
        AnalysisScope.HR_ONLY,
        AnalysisScope.COMPLIANCE_ONLY,
    ]:
        return "domain_summarizer"
    
    # Full analysis - go to risk aggregation
    return "risk_aggregator"


# Export the compiled graph for langgraph.json
graph = build_supervisor_graph()
