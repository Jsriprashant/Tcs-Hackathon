"""Legal Agent MVP Graph - 3 category linear flow.

This module implements the simplified MVP legal agent with:
- Linear flow: init -> litigation -> contracts -> ip -> scoring -> END
- No ReAct loop - deterministic node progression
- LLM-based finding extraction with structured JSON output
"""

import json
from typing import List
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, SystemMessage

from src.legal_agent.state import LegalAgentState, Finding, CategoryScore, LegalResult
from src.legal_agent.utils.retrieval import retrieve_for_category, get_normalized_company_id
from src.legal_agent.utils.scoring import (
    calculate_category_score,
    calculate_total_score,
    determine_risk_level,
    identify_deal_breakers,
)
from src.legal_agent.prompts import (
    SYSTEM_PROMPT,
    LITIGATION_PROMPT,
    CONTRACTS_PROMPT,
    IP_PROMPT,
    COMPANY_NAMES,
)
from src.config.llm_config import get_llm
from src.common.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# LLM ANALYSIS FUNCTIONS
# =============================================================================

def parse_llm_findings(response_content: str, category: str) -> List[Finding]:
    """
    Parse LLM response to extract findings.
    
    Args:
        response_content: Raw LLM response (should be JSON array)
        category: Category to enforce on all findings
    
    Returns:
        List of validated Finding objects
    """
    try:
        content = response_content.strip()
        
        # Handle markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        content = content.strip()
        
        # Parse JSON
        data = json.loads(content)
        
        # Handle case where LLM returns object instead of array
        if isinstance(data, dict):
            data = [data]
        
        # Validate and create Finding objects
        findings = []
        for item in data:
            # Enforce category
            item["category"] = category
            # Ensure required fields have defaults
            item.setdefault("potential_liability", None)
            item.setdefault("recommendation", None)
            findings.append(Finding(**item))
        
        logger.info(f"Parsed {len(findings)} findings for {category}")
        return findings
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {category}: {e}")
        logger.debug(f"Raw content: {response_content[:500]}")
        return []
    except Exception as e:
        logger.warning(f"Error parsing findings for {category}: {e}")
        return []


def analyze_category_with_llm(
    company_docs: str,
    benchmark_docs: str,
    category: str,
    company_id: str,
) -> List[Finding]:
    """
    Use LLM to analyze documents and extract findings.
    
    Args:
        company_docs: Retrieved company documents (Markdown)
        benchmark_docs: Retrieved benchmark documents (Markdown)
        category: Category being analyzed (litigation, contracts, ip)
        company_id: Company identifier
    
    Returns:
        List of Finding objects extracted from analysis
    """
    llm = get_llm(temperature=0.0)
    
    # Select appropriate prompt
    prompts = {
        "litigation": LITIGATION_PROMPT,
        "contracts": CONTRACTS_PROMPT,
        "ip": IP_PROMPT,
    }
    
    prompt = prompts[category].format(
        company_id=company_id,
        company_docs=company_docs if company_docs else "No documents found.",
        benchmark_docs=benchmark_docs if benchmark_docs else "No benchmark documents available.",
    )
    
    system = SYSTEM_PROMPT.format(category=category)
    
    try:
        response = llm.invoke([
            SystemMessage(content=system),
            SystemMessage(content=prompt),
        ])
        
        return parse_llm_findings(response.content, category)
        
    except Exception as e:
        logger.error(f"LLM error in {category} analysis: {e}")
        return []


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def init_node(state: LegalAgentState) -> dict:
    """
    Initialize the analysis with company context.
    
    Normalizes company ID and sets up initial state.
    """
    company_id = get_normalized_company_id(state.company_id) if state.company_id else "UNKNOWN"
    company_name = COMPANY_NAMES.get(company_id, company_id)
    
    logger.info(f"Initializing legal analysis for {company_name} ({company_id})")
    
    return {
        "company_id": company_id,
        "company_name": company_name,
        "current_phase": "litigation",
    }


def litigation_node(state: LegalAgentState) -> dict:
    """
    Analyze litigation exposure.
    
    Retrieves litigation documents and uses LLM to identify risks.
    """
    logger.info(f"Analyzing litigation for {state.company_id}")
    
    # Retrieve documents
    company_docs, benchmark_docs = retrieve_for_category(
        state.company_id, "litigation"
    )
    
    # Analyze with LLM
    findings = analyze_category_with_llm(
        company_docs=company_docs,
        benchmark_docs=benchmark_docs,
        category="litigation",
        company_id=state.company_id,
    )
    
    # Calculate score
    score = calculate_category_score("litigation", findings)
    
    logger.info(f"Litigation: Found {len(findings)} issues, score {score.points_earned}/{score.max_points}")
    
    return {
        "findings": state.findings + findings,
        "category_scores": {**state.category_scores, "litigation": score},
        "current_phase": "contracts",
    }


def contracts_node(state: LegalAgentState) -> dict:
    """
    Analyze contract risks.
    
    Retrieves contract documents and uses LLM to identify risks.
    """
    logger.info(f"Analyzing contracts for {state.company_id}")
    
    company_docs, benchmark_docs = retrieve_for_category(
        state.company_id, "contracts"
    )
    
    findings = analyze_category_with_llm(
        company_docs=company_docs,
        benchmark_docs=benchmark_docs,
        category="contracts",
        company_id=state.company_id,
    )
    
    score = calculate_category_score("contracts", findings)
    
    logger.info(f"Contracts: Found {len(findings)} issues, score {score.points_earned}/{score.max_points}")
    
    return {
        "findings": state.findings + findings,
        "category_scores": {**state.category_scores, "contracts": score},
        "current_phase": "ip",
    }


def ip_node(state: LegalAgentState) -> dict:
    """
    Analyze IP portfolio.
    
    Retrieves IP documents and uses LLM to identify risks.
    """
    logger.info(f"Analyzing IP for {state.company_id}")
    
    company_docs, benchmark_docs = retrieve_for_category(
        state.company_id, "ip"
    )
    
    findings = analyze_category_with_llm(
        company_docs=company_docs,
        benchmark_docs=benchmark_docs,
        category="ip",
        company_id=state.company_id,
    )
    
    score = calculate_category_score("ip", findings)
    
    logger.info(f"IP: Found {len(findings)} issues, score {score.points_earned}/{score.max_points}")
    
    return {
        "findings": state.findings + findings,
        "category_scores": {**state.category_scores, "ip": score},
        "current_phase": "scoring",
    }


def scoring_node(state: LegalAgentState) -> dict:
    """
    Calculate final score and generate result.
    
    Aggregates all category scores and creates the final LegalResult.
    """
    logger.info(f"Calculating final score for {state.company_id}")
    
    # Calculate total score
    total_score = calculate_total_score(state.category_scores)
    
    # Determine risk level
    risk_level = determine_risk_level(total_score)
    
    # Identify deal breakers
    deal_breakers = identify_deal_breakers(state.findings)
    
    # Create final result
    result = LegalResult(
        company_id=state.company_id,
        company_name=state.company_name,
        total_score=total_score,
        risk_level=risk_level,
        category_scores=state.category_scores,
        findings=state.findings,
        deal_breakers=deal_breakers,
        confidence=0.85,
    )
    
    # Create human-readable summary
    summary = create_summary(result)
    
    logger.info(f"Legal analysis complete: {total_score}/100 ({risk_level})")
    
    return {
        "result": result,
        "messages": [AIMessage(content=summary)],
        "current_phase": "complete",
    }


def create_summary(result: LegalResult) -> str:
    """
    Create human-readable summary message.
    
    Args:
        result: The final LegalResult
    
    Returns:
        Formatted markdown summary
    """
    lit_score = result.category_scores.get("litigation")
    con_score = result.category_scores.get("contracts")
    ip_score = result.category_scores.get("ip")
    
    # Format findings list
    findings_summary = ""
    for f in result.findings[:7]:
        severity_icon = {"critical": "!!", "high": "!", "medium": "*", "low": "-"}.get(f.severity, "-")
        findings_summary += f"  [{severity_icon}] [{f.severity.upper()}] {f.title}\n"
    
    if len(result.findings) > 7:
        findings_summary += f"  ... and {len(result.findings) - 7} more findings\n"
    
    # Format deal breakers
    deal_breaker_text = ""
    if result.deal_breakers:
        for db in result.deal_breakers:
            deal_breaker_text += f"  [!!] {db}\n"
    else:
        deal_breaker_text = "  None identified\n"
    
    # Get recommendation based on risk level
    recommendations = {
        "LOW": "Proceed with standard terms and conditions.",
        "MODERATE": "Proceed with enhanced representations and warranties.",
        "HIGH": "Significant negotiation required before proceeding. Consider escrow provisions.",
        "CRITICAL": "Consider deal restructure or walk away. Major legal risks identified.",
    }
    recommendation = recommendations.get(result.risk_level, "Further analysis required.")
    
    # Build summary
    summary = f"""
# Legal Due Diligence Report: {result.company_name}

## Executive Summary
- **Company:** {result.company_name} ({result.company_id})
- **Total Score:** {result.total_score}/100
- **Risk Level:** {result.risk_level}
- **Confidence:** {result.confidence:.0%}

## Category Breakdown

| Category | Score | Max | Status |
|----------|-------|-----|--------|
| Litigation Exposure | {lit_score.points_earned if lit_score else 0} | 35 | {"HIGH RISK" if lit_score and lit_score.points_earned < 25 else "OK"} |
| Contract Risk | {con_score.points_earned if con_score else 0} | 35 | {"HIGH RISK" if con_score and con_score.points_earned < 25 else "OK"} |
| IP Portfolio | {ip_score.points_earned if ip_score else 0} | 30 | {"HIGH RISK" if ip_score and ip_score.points_earned < 20 else "OK"} |

## Key Findings ({len(result.findings)} total)

{findings_summary}
## Deal Breakers

{deal_breaker_text}
## Recommendation

{recommendation}

---
*Report generated by Legal Due Diligence Agent (MVP)*
*Analysis based on {len(result.findings)} identified issues across 3 categories*
"""
    return summary


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_legal_agent_graph():
    """
    Build the MVP legal agent graph.
    
    Flow:
        init -> litigation -> contracts -> ip -> scoring -> END
    
    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(LegalAgentState)
    
    # Add nodes
    workflow.add_node("init", init_node)
    workflow.add_node("litigation", litigation_node)
    workflow.add_node("contracts", contracts_node)
    workflow.add_node("ip", ip_node)
    workflow.add_node("scoring", scoring_node)
    
    # Set entry point
    workflow.set_entry_point("init")
    
    # Define linear edges
    workflow.add_edge("init", "litigation")
    workflow.add_edge("litigation", "contracts")
    workflow.add_edge("contracts", "ip")
    workflow.add_edge("ip", "scoring")
    workflow.add_edge("scoring", END)
    
    return workflow.compile()


# Create compiled graph instance
legal_agent = build_legal_agent_graph()

# Alias for langgraph.json
graph = legal_agent
