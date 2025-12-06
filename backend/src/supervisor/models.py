"""
Enhanced models for Supervisor Agent M&A analysis.

This module contains Pydantic models for:
- Agent outputs with structured findings
- Domain risk scores with weighted calculations
- Aggregated risk assessment
- Deal analysis with reasoning chain
"""

from typing import Optional, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# ANALYSIS SCOPE & DEAL TYPE ENUMS
# =============================================================================

class AnalysisScope(str, Enum):
    """Granular analysis scope detection."""
    FULL_DUE_DILIGENCE = "FULL_DUE_DILIGENCE"       # Complete M&A analysis
    FINANCIAL_ONLY = "FINANCIAL_ONLY"               # Only financial analysis
    LEGAL_ONLY = "LEGAL_ONLY"                       # Only legal analysis
    HR_ONLY = "HR_ONLY"                             # Only HR analysis
    COMPLIANCE_ONLY = "COMPLIANCE_ONLY"             # Only compliance analysis
    STRATEGIC_ONLY = "STRATEGIC_ONLY"               # Only strategic/synergy analysis
    RISK_ASSESSMENT = "RISK_ASSESSMENT"             # Risk-focused analysis
    COMPARISON = "COMPARISON"                       # Compare multiple targets
    QUICK_OVERVIEW = "QUICK_OVERVIEW"               # High-level summary only


class DealType(str, Enum):
    """Type of M&A transaction."""
    ACQUISITION = "acquisition"
    MERGER = "merger"
    DIVESTITURE = "divestiture"
    JOINT_VENTURE = "joint_venture"
    ASSET_PURCHASE = "asset_purchase"


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Recommendation(str, Enum):
    """Deal recommendation types."""
    GO = "GO"
    NO_GO = "NO_GO"
    CONDITIONAL = "CONDITIONAL"


# =============================================================================
# FINDING & RISK FACTOR MODELS
# =============================================================================

class Finding(BaseModel):
    """Individual finding from agent analysis."""
    
    category: str = Field(description="Category of finding (e.g., 'revenue', 'litigation', 'attrition')")
    title: str = Field(description="Brief title of the finding")
    description: str = Field(description="Detailed description of the finding")
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")
    impact: Optional[str] = Field(default=None, description="Potential impact on the deal")
    data_points: list[str] = Field(default_factory=list, description="Supporting data points")
    source: Optional[str] = Field(default=None, description="Source document or data")
    
    class Config:
        extra = "allow"


class RiskFactor(BaseModel):
    """Individual risk factor identified."""
    
    factor_id: str = Field(description="Unique identifier for the risk factor")
    name: str = Field(description="Name of the risk factor")
    description: str = Field(description="Detailed description")
    severity: Literal["low", "medium", "high", "critical"]
    probability: float = Field(ge=0.0, le=1.0, description="Probability of risk materializing")
    impact_score: float = Field(ge=0.0, le=1.0, description="Impact if risk materializes")
    is_deal_breaker: bool = Field(default=False, description="Whether this is a potential deal-breaker")
    mitigation: Optional[str] = Field(default=None, description="Suggested mitigation")
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score as probability × impact."""
        return self.probability * self.impact_score


# =============================================================================
# AGENT OUTPUT MODEL
# =============================================================================

class AgentOutput(BaseModel):
    """Structured output from a domain agent."""
    
    agent_name: str = Field(description="Name of the agent (e.g., 'finance_agent')")
    domain: str = Field(description="Domain analyzed (e.g., 'finance', 'legal', 'hr')")
    
    # Summary
    summary: str = Field(description="Executive summary of agent's analysis")
    
    # Findings
    findings: list[Finding] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list, description="Top 3-5 key findings")
    
    # Risk Assessment
    risk_score: float = Field(ge=0.0, le=1.0, description="Domain risk score 0-1")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    
    # Recommendations
    recommendations: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list, description="Critical red flags identified")
    positive_factors: list[str] = Field(default_factory=list, description="Positive aspects found")
    
    # Metadata
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    data_quality: Literal["high", "medium", "low"] = Field(default="medium")
    documents_analyzed: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Raw response for reference
    raw_response: Optional[str] = Field(default=None, description="Raw LLM response if needed")
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# DOMAIN RISK SCORE MODEL
# =============================================================================

class DomainRiskScore(BaseModel):
    """Risk score for a specific domain with weighting."""
    
    domain: str = Field(description="Domain name (finance, legal, hr, compliance)")
    score: float = Field(ge=0.0, le=1.0, description="Risk score 0-1")
    weight: float = Field(ge=0.0, le=1.0, default=0.25, description="Weight in overall calculation")
    
    # Contributing factors
    contributing_factors: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)
    
    # Weighted score
    @property
    def weighted_score(self) -> float:
        """Calculate weighted contribution to overall risk."""
        return self.score * self.weight
    
    # Risk level derived from score
    @property
    def risk_level(self) -> RiskLevel:
        if self.score < 0.3:
            return RiskLevel.LOW
        elif self.score < 0.5:
            return RiskLevel.MEDIUM
        elif self.score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


# Domain weight configuration
DOMAIN_WEIGHTS = {
    "finance": 0.35,
    "legal": 0.25,
    "hr": 0.20,
    "compliance": 0.20,
    "strategic": 0.15,  # When included
}


# =============================================================================
# AGGREGATED RISK MODEL
# =============================================================================

class AggregatedRisk(BaseModel):
    """Aggregated risk assessment across all domains."""
    
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted overall risk score")
    risk_level: RiskLevel = Field(description="Overall risk level classification")
    
    # Key factors
    deal_breakers: list[str] = Field(default_factory=list, description="Identified deal-breakers")
    key_concerns: list[str] = Field(default_factory=list, description="Top concerns across domains")
    positive_factors: list[str] = Field(default_factory=list, description="Positive factors across domains")
    
    # Domain breakdown
    domain_scores: dict[str, float] = Field(default_factory=dict, description="Score by domain")
    highest_risk_domain: Optional[str] = Field(default=None, description="Domain with highest risk")
    lowest_risk_domain: Optional[str] = Field(default=None, description="Domain with lowest risk")
    
    # Adjustments
    deal_breaker_penalty_applied: bool = Field(default=False)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    
    @classmethod
    def from_domain_scores(cls, domain_risk_scores: dict[str, "DomainRiskScore"]) -> "AggregatedRisk":
        """Calculate aggregated risk from domain scores."""
        if not domain_risk_scores:
            return cls(
                overall_score=0.5,
                risk_level=RiskLevel.MEDIUM,
                confidence=0.5
            )
        
        # Calculate weighted average
        total_weight = sum(d.weight for d in domain_risk_scores.values())
        weighted_sum = sum(d.weighted_score for d in domain_risk_scores.values())
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Determine risk level
        if overall_score < 0.3:
            risk_level = RiskLevel.LOW
        elif overall_score < 0.5:
            risk_level = RiskLevel.MEDIUM
        elif overall_score < 0.7:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        # Find highest/lowest risk domains
        domain_scores = {d.domain: d.score for d in domain_risk_scores.values()}
        highest_risk = max(domain_scores, key=domain_scores.get) if domain_scores else None
        lowest_risk = min(domain_scores, key=domain_scores.get) if domain_scores else None
        
        return cls(
            overall_score=overall_score,
            risk_level=risk_level,
            domain_scores=domain_scores,
            highest_risk_domain=highest_risk,
            lowest_risk_domain=lowest_risk
        )


# =============================================================================
# REASONING STEP MODEL
# =============================================================================

class ReasoningStep(BaseModel):
    """Single step in the reasoning chain."""
    
    step_number: int = Field(description="Step number in the chain")
    analysis: str = Field(description="What was analyzed")
    finding: str = Field(description="What was found")
    implication: str = Field(description="Risk/deal implication")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


# =============================================================================
# DEAL ANALYSIS MODEL
# =============================================================================

class DealAnalysis(BaseModel):
    """Complete deal analysis with recommendation."""
    
    # Core recommendation
    recommendation: Recommendation = Field(description="GO / NO_GO / CONDITIONAL")
    recommendation_confidence: float = Field(ge=0.0, le=1.0)
    
    # Executive summary
    executive_summary: str = Field(description="One-paragraph deal summary")
    
    # Reasoning chain
    reasoning_chain: list[ReasoningStep] = Field(default_factory=list)
    key_reasoning_points: list[str] = Field(default_factory=list, description="Top 5 reasoning points")
    
    # Risk summary
    aggregated_risk: Optional[AggregatedRisk] = None
    
    # Deal structure recommendations
    valuation_impact: Optional[str] = Field(default=None, description="Impact on valuation")
    suggested_price_adjustment: Optional[str] = Field(default=None, description="Suggested price adjustment %")
    key_terms_required: list[str] = Field(default_factory=list, description="Required deal terms")
    
    # Integration considerations
    integration_complexity: Literal["low", "medium", "high"] = Field(default="medium")
    integration_considerations: list[str] = Field(default_factory=list)
    integration_timeline: Optional[str] = Field(default=None, description="Estimated integration timeline")
    
    # Next steps
    immediate_actions: list[str] = Field(default_factory=list)
    additional_dd_required: list[str] = Field(default_factory=list, description="Additional due diligence needed")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    agents_consulted: list[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# ANALYSIS PLAN MODEL
# =============================================================================

class AnalysisPlan(BaseModel):
    """Execution plan for M&A analysis."""
    
    plan_id: str = Field(description="Unique plan identifier")
    analysis_scope: AnalysisScope = Field(description="Scope of analysis requested")
    
    # Agents to invoke
    required_agents: list[str] = Field(default_factory=list)
    optional_agents: list[str] = Field(default_factory=list)
    
    # Execution strategy
    execution_mode: Literal["sequential", "parallel", "hybrid"] = Field(default="hybrid")
    agent_order: list[list[str]] = Field(default_factory=list, description="Ordered list of agent groups")
    
    # Dependencies
    dependencies: dict[str, list[str]] = Field(default_factory=dict, description="Agent dependencies")
    
    # Output requirements
    require_risk_score: bool = Field(default=True)
    require_recommendation: bool = Field(default=True)
    report_format: Literal["full", "summary", "executive"] = Field(default="full")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    estimated_duration_seconds: Optional[int] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# ENHANCED INTENT RESULT
# =============================================================================

class EnhancedIntentResult(BaseModel):
    """Enhanced intent classification with analysis scope."""
    
    # Basic intent (from original IntentType)
    intent: str = Field(description="Basic intent type")
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Company extraction
    acquirer_company: Optional[str] = Field(default=None)
    target_company: Optional[str] = Field(default=None)
    additional_companies: list[str] = Field(default_factory=list, description="For comparisons")
    
    # Analysis scope
    analysis_scope: AnalysisScope = Field(default=AnalysisScope.FULL_DUE_DILIGENCE)
    required_domains: list[str] = Field(default_factory=list, description="Required analysis domains")
    
    # Deal context
    deal_type: DealType = Field(default=DealType.ACQUISITION)
    deal_context: Optional[str] = Field(default=None, description="Additional context from query")
    
    # Execution hints
    priority_domain: Optional[str] = Field(default=None, description="User's primary focus")
    depth: Literal["quick", "standard", "deep"] = Field(default="standard")
    
    # Chain activation
    should_activate_chain: bool = Field(default=False)
    reasoning: str = Field(default="", description="Classification reasoning")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_risk_level(score: float) -> RiskLevel:
    """Convert numeric score to risk level."""
    if score < 0.3:
        return RiskLevel.LOW
    elif score < 0.5:
        return RiskLevel.MEDIUM
    elif score < 0.7:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


def get_recommendation_from_risk(risk_score: float, has_deal_breakers: bool = False) -> Recommendation:
    """Determine recommendation based on risk score."""
    if has_deal_breakers:
        return Recommendation.NO_GO
    
    if risk_score < 0.3:
        return Recommendation.GO
    elif risk_score < 0.5:
        return Recommendation.CONDITIONAL  # GO with conditions
    elif risk_score < 0.7:
        return Recommendation.CONDITIONAL  # Significant work needed
    else:
        return Recommendation.NO_GO


def generate_plan_id() -> str:
    """Generate unique plan ID."""
    from datetime import datetime
    import uuid
    return f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
