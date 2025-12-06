"""
Agent Parsers for Supervisor Integration.

This module provides input adapters and output parsers for transforming data
between the supervisor state and specialized agents.

Design Principles:
- No fallbacks or workarounds - strict type enforcement
- Bidirectional transformation: SupervisorState ↔ AgentState
- Structured output extraction into AgentOutput model
"""

import json
import re
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, AIMessage

from src.supervisor.state import SupervisorState
from src.supervisor.models import (
    AgentOutput,
    Finding,
    RiskFactor,
    RiskLevel,
)
from src.common.state import CompanyInfo
from src.common.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# LEGAL AGENT PARSERS
# =============================================================================

def create_legal_agent_input(state: SupervisorState) -> Dict[str, Any]:
    """
    Transform supervisor state into legal agent input format.
    
    Legal Agent expects:
    {
        "messages": [],  # Can be empty
        "company_id": "BBD"  # Required - company identifier
    }
    
    Args:
        state: Current supervisor state with target company info
        
    Returns:
        Dict formatted for legal_agent.invoke()
        
    Raises:
        ValueError: If no target company is specified
    """
    if not state.target:
        raise ValueError("Legal agent requires target company to be specified")
    
    # Extract company_id from CompanyInfo
    company_id = state.target.company_id.upper()
    
    logger.info(f"Creating legal agent input for company_id: {company_id}")
    
    return {
        "messages": [],  # Legal agent doesn't need prior messages
        "company_id": company_id,
    }


def parse_legal_agent_output(result: Dict[str, Any]) -> AgentOutput:
    """
    Parse legal agent output into structured AgentOutput.
    
    Legal Agent returns:
    {
        "result": LegalResult,  # Structured result with scores
        "findings": [...],
        "category_scores": {...},
        "messages": [...]
    }
    
    Args:
        result: Raw output from legal_agent.invoke()
        
    Returns:
        AgentOutput with standardized structure
    """
    legal_result = result.get("result")
    
    if not legal_result:
        logger.warning("Legal agent returned no structured result")
        return AgentOutput(
            agent_name="legal_agent",
            domain="legal",
            summary="Legal analysis completed but no structured result available.",
            risk_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.5,
            data_quality="low",
        )
    
    # Convert legal findings to AgentOutput findings
    findings = []
    for f in legal_result.findings:
        findings.append(Finding(
            category=f.category,
            title=f.title,
            description=f.description,
            severity=f.severity,
            impact=f.recommendation,
            source=f.source_document,
            data_points=[f"Potential liability: ${f.potential_liability:,.0f}" if f.potential_liability else "Liability: Unknown"],
        ))
    
    # Convert severity findings to risk factors
    risk_factors = []
    for i, f in enumerate(legal_result.findings):
        is_deal_breaker = f.severity == "critical" or f.title in legal_result.deal_breakers
        
        # Map severity to probability/impact
        severity_map = {
            "critical": (0.9, 0.95),
            "high": (0.7, 0.8),
            "medium": (0.5, 0.5),
            "low": (0.3, 0.3),
        }
        prob, impact = severity_map.get(f.severity, (0.5, 0.5))
        
        risk_factors.append(RiskFactor(
            factor_id=f"legal_{i+1}",
            name=f.title,
            description=f.description,
            severity=f.severity,
            probability=prob,
            impact_score=impact,
            is_deal_breaker=is_deal_breaker,
            mitigation=f.recommendation,
        ))
    
    # Convert legal score (0-100) to risk score (0-1)
    # High score = low risk, so invert
    legal_score = legal_result.total_score
    risk_score = 1.0 - (legal_score / 100.0)
    
    # Map risk level
    risk_level_map = {
        "LOW": RiskLevel.LOW,
        "MODERATE": RiskLevel.MEDIUM,
        "HIGH": RiskLevel.HIGH,
        "CRITICAL": RiskLevel.CRITICAL,
    }
    risk_level = risk_level_map.get(legal_result.risk_level, RiskLevel.MEDIUM)
    
    # Extract key findings (top 5)
    key_findings = [f"[{f.severity.upper()}] {f.title}" for f in legal_result.findings[:5]]
    
    # Create summary
    summary = f"""Legal Due Diligence Score: {legal_score}/100 ({legal_result.risk_level})

Category Scores:
- Litigation: {legal_result.category_scores.get('litigation', {}).points_earned if hasattr(legal_result.category_scores.get('litigation', {}), 'points_earned') else 'N/A'}/35
- Contracts: {legal_result.category_scores.get('contracts', {}).points_earned if hasattr(legal_result.category_scores.get('contracts', {}), 'points_earned') else 'N/A'}/35
- IP Portfolio: {legal_result.category_scores.get('ip', {}).points_earned if hasattr(legal_result.category_scores.get('ip', {}), 'points_earned') else 'N/A'}/30

Total Findings: {len(legal_result.findings)}
Deal Breakers: {len(legal_result.deal_breakers)}"""
    
    # Get last message for raw response
    messages = result.get("messages", [])
    raw_response = messages[-1].content if messages else None
    
    return AgentOutput(
        agent_name="legal_agent",
        domain="legal",
        summary=summary,
        findings=findings,
        key_findings=key_findings,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_factors=risk_factors,
        recommendations=[f.recommendation for f in legal_result.findings if f.recommendation],
        red_flags=[f.title for f in legal_result.findings if f.severity in ("critical", "high")],
        positive_factors=["No deal breakers identified"] if not legal_result.deal_breakers else [],
        confidence=legal_result.confidence,
        data_quality="high",
        raw_response=raw_response,
    )


# =============================================================================
# FINANCE AGENT PARSERS
# =============================================================================

def create_finance_agent_input(state: SupervisorState) -> Dict[str, Any]:
    """
    Transform supervisor state into finance agent input format.
    
    Finance Agent expects (FinanceAgentState):
    {
        "messages": [HumanMessage(content="Analyze BBD...")],
        "deal_type": "acquisition",
        "analysis_request": {  # Optional
            "company_id": "BBD",
            "analysis_types": [...],
            "years_to_analyze": 5
        }
    }
    
    Args:
        state: Current supervisor state with target company info
        
    Returns:
        Dict formatted for finance_agent.invoke()
        
    Raises:
        ValueError: If no target company is specified
    """
    from langchain_core.messages import HumanMessage
    
    if not state.target:
        raise ValueError("Finance agent requires target company to be specified")
    
    company_id = state.target.company_id.upper()
    company_name = state.target.company_name
    
    logger.info(f"Creating finance agent input for company_id: {company_id}")
    
    # Create analysis prompt for the finance agent
    analysis_prompt = f"""Analyze the financial health and M&A readiness of {company_name} ({company_id}).

Please perform a comprehensive financial due diligence analysis including:
1. Retrieve financial documents for {company_id}
2. Calculate key financial ratios
3. Identify any red flags or concerns
4. Calculate the TCS M&A Financial Score
5. Provide a recommendation (GO/NO-GO/CONDITIONAL)

Company ID: {company_id}
Deal Type: {state.deal_type or 'acquisition'}"""
    
    return {
        "messages": [HumanMessage(content=analysis_prompt)],
        "deal_type": state.deal_type or "acquisition",
    }


def parse_finance_agent_output(result: Dict[str, Any]) -> AgentOutput:
    """
    Parse finance agent output into structured AgentOutput.
    
    Finance Agent returns ReAct-style output with tool calls.
    The structured data is in the messages from tool responses.
    
    Args:
        result: Raw output from finance_agent.invoke()
        
    Returns:
        AgentOutput with standardized structure
    """
    messages = result.get("messages", [])
    
    if not messages:
        logger.warning("Finance agent returned no messages")
        return AgentOutput(
            agent_name="finance_agent",
            domain="finance",
            summary="Financial analysis completed but no output available.",
            risk_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.5,
            data_quality="low",
        )
    
    # Extract data from messages
    tcs_score = None
    ratios = {}
    red_flags = []
    recommendation = None
    raw_response = ""
    
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        # Try to parse TCS score from tool responses
        if "tcs_score" in content.lower():
            try:
                # Find JSON in content
                json_match = re.search(r'\{[^{}]*"tcs_score"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    score_data = json.loads(json_match.group())
                    tcs_score = score_data.get("tcs_score", {})
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Try to parse calculated ratios
        if "calculated_ratios" in content.lower():
            try:
                json_match = re.search(r'\{[^{}]*"calculated_ratios"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    ratios_data = json.loads(json_match.group())
                    ratios = ratios_data.get("calculated_ratios", {})
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Collect final response
        if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
            raw_response = content
    
    # Extract structured data from tcs_score if available
    if tcs_score:
        total_score = tcs_score.get("total", 50)
        interpretation = tcs_score.get("interpretation", "Moderate")
        
        # Convert TCS score (0-100) to risk score (0-1)
        # High TCS score = low risk
        risk_score = 1.0 - (total_score / 100.0)
        
        # Determine risk level based on TCS interpretation
        if interpretation in ["Strong", "Very Strong"]:
            risk_level = RiskLevel.LOW
        elif interpretation in ["Moderate"]:
            risk_level = RiskLevel.MEDIUM
        elif interpretation in ["Weak", "High Risk"]:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
    else:
        # Fallback: estimate from message content
        risk_score = estimate_risk_from_content(raw_response)
        risk_level = calculate_risk_level_from_score(risk_score)
        total_score = int((1 - risk_score) * 100)
        interpretation = "Estimated"
    
    # Build findings from ratios and red flags
    findings = []
    key_findings = []
    
    # Add ratio findings
    for ratio_name, value in ratios.items():
        if value != "N/A" and value is not None:
            severity = "low"
            # Determine severity based on ratio thresholds
            if ratio_name in ["debt_to_equity", "debt_to_assets"]:
                if isinstance(value, (int, float)) and value > 2.0:
                    severity = "high"
                    key_findings.append(f"[HIGH] High {ratio_name}: {value}")
            elif ratio_name in ["current_ratio", "quick_ratio"]:
                if isinstance(value, (int, float)) and value < 1.0:
                    severity = "high"
                    key_findings.append(f"[HIGH] Low {ratio_name}: {value}")
            
            findings.append(Finding(
                category="financial_ratios",
                title=ratio_name.replace("_", " ").title(),
                description=f"Calculated {ratio_name}: {value}",
                severity=severity,
            ))
    
    # Build summary
    summary = f"""TCS Financial Score: {total_score}/100 ({interpretation})

Key Ratios:
- Gross Margin: {ratios.get('gross_profit_margin', 'N/A')}%
- Net Margin: {ratios.get('net_profit_margin', 'N/A')}%
- Current Ratio: {ratios.get('current_ratio', 'N/A')}
- Debt/Equity: {ratios.get('debt_to_equity', 'N/A')}
- ROE: {ratios.get('roe', 'N/A')}%

Red Flags: {len(red_flags) if red_flags else 'None identified'}"""
    
    # Extract recommendations from final message
    recommendations = []
    if "PROCEED" in raw_response.upper() or "GO" in raw_response.upper():
        recommendations.append("Proceed with acquisition")
    elif "CAUTION" in raw_response.upper():
        recommendations.append("Proceed with caution - negotiate valuation discount")
    elif "REJECT" in raw_response.upper():
        recommendations.append("Consider rejecting - significant financial risks")
    
    return AgentOutput(
        agent_name="finance_agent",
        domain="finance",
        summary=summary,
        findings=findings,
        key_findings=key_findings[:5],
        risk_score=risk_score,
        risk_level=risk_level,
        risk_factors=[],  # Finance agent doesn't produce explicit risk factors
        recommendations=recommendations,
        red_flags=red_flags,
        positive_factors=[],
        confidence=0.8,
        data_quality="medium",
        raw_response=raw_response[:2000] if raw_response else None,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_risk_from_content(content: str) -> float:
    """Estimate risk score from message content analysis."""
    if not content:
        return 0.5
    
    content_lower = content.lower()
    
    # High risk indicators
    high_risk_keywords = ["reject", "critical", "deal breaker", "high risk", "red flag", "concern"]
    high_risk_count = sum(1 for kw in high_risk_keywords if kw in content_lower)
    
    # Low risk indicators
    low_risk_keywords = ["strong", "healthy", "proceed", "recommend", "positive", "stable"]
    low_risk_count = sum(1 for kw in low_risk_keywords if kw in content_lower)
    
    # Calculate score
    if high_risk_count > low_risk_count:
        return 0.6 + (high_risk_count * 0.05)
    elif low_risk_count > high_risk_count:
        return 0.4 - (low_risk_count * 0.05)
    else:
        return 0.5


def calculate_risk_level_from_score(score: float) -> RiskLevel:
    """Convert risk score to risk level."""
    if score >= 0.7:
        return RiskLevel.HIGH
    elif score >= 0.5:
        return RiskLevel.MEDIUM
    elif score >= 0.3:
        return RiskLevel.LOW
    else:
        return RiskLevel.LOW


# =============================================================================
# CONSOLIDATED OUTPUT MODEL
# =============================================================================

class ConsolidatedResult:
    """
    Consolidated result from all agents for frontend display.
    
    Provides structured data for:
    - Executive summary table
    - Domain-wise scoring with color coding
    - Key findings across all domains
    - Overall recommendation
    """
    
    def __init__(
        self,
        company_id: str,
        company_name: str,
        finance_output: Optional[AgentOutput] = None,
        legal_output: Optional[AgentOutput] = None,
        hr_output: Optional[AgentOutput] = None,
    ):
        self.company_id = company_id
        self.company_name = company_name
        self.finance_output = finance_output
        self.legal_output = legal_output
        self.hr_output = hr_output
    
    def get_scoring_table(self) -> List[Dict[str, Any]]:
        """
        Generate scoring table for frontend with color coding.
        
        Returns:
            List of dicts with domain, score, status, color
        """
        table = []
        
        # Define color thresholds
        def get_color(risk_score: float) -> str:
            """Map risk score to color: lower risk = green."""
            if risk_score <= 0.3:
                return "green"
            elif risk_score <= 0.5:
                return "yellow"
            elif risk_score <= 0.7:
                return "orange"
            else:
                return "red"
        
        def get_status(risk_level: RiskLevel) -> str:
            """Map risk level to status text."""
            return {
                RiskLevel.LOW: "LOW RISK",
                RiskLevel.MEDIUM: "MODERATE",
                RiskLevel.HIGH: "HIGH RISK",
                RiskLevel.CRITICAL: "CRITICAL",
            }.get(risk_level, "UNKNOWN")
        
        # Finance row
        if self.finance_output:
            # Convert risk score to health score (inverse)
            health_score = int((1 - self.finance_output.risk_score) * 100)
            table.append({
                "domain": "Financial",
                "agent": "finance_agent",
                "score": health_score,
                "max_score": 100,
                "risk_score": round(self.finance_output.risk_score, 2),
                "status": get_status(self.finance_output.risk_level),
                "color": get_color(self.finance_output.risk_score),
                "key_findings": self.finance_output.key_findings[:3],
                "confidence": self.finance_output.confidence,
            })
        
        # Legal row
        if self.legal_output:
            health_score = int((1 - self.legal_output.risk_score) * 100)
            table.append({
                "domain": "Legal",
                "agent": "legal_agent",
                "score": health_score,
                "max_score": 100,
                "risk_score": round(self.legal_output.risk_score, 2),
                "status": get_status(self.legal_output.risk_level),
                "color": get_color(self.legal_output.risk_score),
                "key_findings": self.legal_output.key_findings[:3],
                "confidence": self.legal_output.confidence,
            })
        
        # HR row (placeholder for when HR is integrated)
        if self.hr_output:
            health_score = int((1 - self.hr_output.risk_score) * 100)
            table.append({
                "domain": "HR",
                "agent": "hr_agent",
                "score": health_score,
                "max_score": 100,
                "risk_score": round(self.hr_output.risk_score, 2),
                "status": get_status(self.hr_output.risk_level),
                "color": get_color(self.hr_output.risk_score),
                "key_findings": self.hr_output.key_findings[:3],
                "confidence": self.hr_output.confidence,
            })
        
        return table
    
    def get_overall_score(self) -> Dict[str, Any]:
        """
        Calculate overall weighted score.
        
        Weights: Finance 40%, Legal 35%, HR 25%
        """
        weights = {"finance": 0.40, "legal": 0.35, "hr": 0.25}
        total_weight = 0
        weighted_risk = 0
        
        if self.finance_output:
            weighted_risk += self.finance_output.risk_score * weights["finance"]
            total_weight += weights["finance"]
        
        if self.legal_output:
            weighted_risk += self.legal_output.risk_score * weights["legal"]
            total_weight += weights["legal"]
        
        if self.hr_output:
            weighted_risk += self.hr_output.risk_score * weights["hr"]
            total_weight += weights["hr"]
        
        if total_weight > 0:
            overall_risk = weighted_risk / total_weight
            overall_health = int((1 - overall_risk) * 100)
        else:
            overall_risk = 0.5
            overall_health = 50
        
        # Determine recommendation
        if overall_risk <= 0.3:
            recommendation = "GO"
            recommendation_color = "green"
        elif overall_risk <= 0.5:
            recommendation = "CONDITIONAL"
            recommendation_color = "yellow"
        elif overall_risk <= 0.7:
            recommendation = "CAUTION"
            recommendation_color = "orange"
        else:
            recommendation = "NO-GO"
            recommendation_color = "red"
        
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "overall_health_score": overall_health,
            "overall_risk_score": round(overall_risk, 2),
            "recommendation": recommendation,
            "recommendation_color": recommendation_color,
            "domains_analyzed": sum([
                1 if self.finance_output else 0,
                1 if self.legal_output else 0,
                1 if self.hr_output else 0,
            ]),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "company": {
                "id": self.company_id,
                "name": self.company_name,
            },
            "overall": self.get_overall_score(),
            "scoring_table": self.get_scoring_table(),
            "domain_details": {
                "finance": self.finance_output.model_dump() if self.finance_output else None,
                "legal": self.legal_output.model_dump() if self.legal_output else None,
                "hr": self.hr_output.model_dump() if self.hr_output else None,
            },
        }
