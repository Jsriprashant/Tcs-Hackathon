"""State definitions for Supervisor Agent - Enhanced v2.0."""

from typing import Optional, Literal, Annotated, Dict, List
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from src.common.state import BaseAgentState, RiskScore, AnalysisResult, CompanyInfo
from src.common.intent_classifier import IntentType

# Import enhanced models
from src.supervisor.models import (
    AnalysisScope,
    DealType,
    EnhancedIntentResult,
    AnalysisPlan,
    AgentOutput,
    DomainRiskScore,
    AggregatedRisk,
    DealAnalysis,
    RiskLevel,
    Recommendation,
)


class SupervisorState(BaseAgentState):
    """
    Enhanced State for the Supervisor Agent (v2.0).
    
    The supervisor maintains the overall state of the due diligence process
    and coordinates between specialized agents.
    
    ENHANCED FEATURES:
    - Granular analysis scope tracking
    - Structured agent outputs
    - Risk aggregation with weighted scoring
    - Deal analysis with reasoning chain
    - Dynamic execution planning
    """
    
    # Conversation
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    
    # ==========================================================================
    # INTENT CLASSIFICATION (Enhanced)
    # ==========================================================================
    intent_classified: bool = Field(
        default=False, 
        description="Whether intent classification has been performed"
    )
    intent_type: Optional[Literal[
        "MA_DUE_DILIGENCE",
        "MA_QUESTION",
        "INFORMATIONAL",
        "GREETING",
        "HELP",
        "UNKNOWN"
    ]] = Field(
        default=None,
        description="Classified intent type from user query"
    )
    intent_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of intent classification"
    )
    chain_activated: bool = Field(
        default=False,
        description="Whether the full agent chain should run"
    )
    
    # Enhanced intent result (NEW)
    enhanced_intent: Optional[EnhancedIntentResult] = Field(
        default=None,
        description="Full enhanced intent classification result"
    )
    
    # ==========================================================================
    # ANALYSIS PLANNING (NEW)
    # ==========================================================================
    analysis_plan: Optional[AnalysisPlan] = Field(
        default=None,
        description="Execution plan for the analysis"
    )
    analysis_scope: Optional[AnalysisScope] = Field(
        default=None,
        description="Scope of analysis requested"
    )
    required_domains: List[str] = Field(
        default_factory=list,
        description="Required analysis domains"
    )
    
    # ==========================================================================
    # DEAL CONTEXT
    # ==========================================================================
    deal_id: Optional[str] = None
    deal_type: Literal["merger", "acquisition", "divestiture", "joint_venture", "asset_purchase"] = "acquisition"
    analysis_type: Literal["horizontal", "vertical"] = "horizontal"
    
    # Companies
    acquirer: Optional[CompanyInfo] = None
    target: Optional[CompanyInfo] = None
    additional_targets: List[CompanyInfo] = Field(
        default_factory=list,
        description="Additional targets for comparison analysis"
    )
    
    # ==========================================================================
    # AGENT ROUTING
    # ==========================================================================
    next_agent: Optional[Literal[
        "finance_agent",
        "legal_agent", 
        "hr_agent",
        "analyst_agent",
        "rag_agent",
        "risk_aggregator",
        "master_analyst",
        "domain_summarizer",
        "human",
        "FINISH"
    ]] = None
    
    next_agents: List[str] = Field(
        default_factory=list,
        description="Multiple agents for parallel execution"
    )
    
    # ==========================================================================
    # AGENT OUTPUTS (Enhanced - Structured)
    # ==========================================================================
    # Legacy fields (kept for backward compatibility)
    finance_result: Optional[AnalysisResult] = None
    legal_result: Optional[AnalysisResult] = None
    hr_result: Optional[AnalysisResult] = None
    analyst_result: Optional[AnalysisResult] = None
    
    # New structured outputs (v2.0)
    agent_outputs: Dict[str, AgentOutput] = Field(
        default_factory=dict,
        description="Structured outputs from each agent keyed by agent name"
    )
    
    # ==========================================================================
    # RISK AGGREGATION (Enhanced)
    # ==========================================================================
    # Legacy risk scores
    risk_scores: dict[str, RiskScore] = Field(default_factory=dict)
    
    # New domain risk scores (v2.0)
    domain_risk_scores: Dict[str, DomainRiskScore] = Field(
        default_factory=dict,
        description="Risk scores by domain with weighting"
    )
    
    # Aggregated risk assessment (v2.0)
    aggregated_risk: Optional[AggregatedRisk] = Field(
        default=None,
        description="Aggregated risk across all domains"
    )
    
    # Overall assessment
    overall_risk_score: Optional[float] = None
    overall_risk_level: Optional[RiskLevel] = None
    deal_recommendation: Optional[Literal["GO", "NO_GO", "CONDITIONAL"]] = None
    recommendation_rationale: Optional[str] = None
    
    # ==========================================================================
    # DEAL ANALYSIS (NEW)
    # ==========================================================================
    deal_analysis: Optional[DealAnalysis] = Field(
        default=None,
        description="Complete deal analysis with reasoning chain"
    )
    
    # ==========================================================================
    # EXECUTION TRACKING (Enhanced)
    # ==========================================================================
    agents_invoked: list[str] = Field(default_factory=list)
    agents_pending: List[str] = Field(default_factory=list)
    agents_completed: List[str] = Field(default_factory=list)
    agents_failed: List[str] = Field(default_factory=list)
    
    current_phase: Literal[
        "initialization",
        "intent_classification",
        "planning",
        "document_retrieval", 
        "domain_analysis",
        "financial_analysis",
        "legal_analysis",
        "hr_analysis",
        "strategic_analysis",
        "risk_aggregation",
        "recommendation_generation",
        "report_formatting",
        "consolidation",
        "complete"
    ] = "initialization"
    
    execution_phase: Literal[
        "intent_classification",
        "planning",
        "document_retrieval",
        "domain_analysis",
        "risk_aggregation",
        "recommendation_generation",
        "report_formatting",
        "complete"
    ] = Field(
        default="intent_classification",
        description="Current execution phase in the plan"
    )
    
    # ==========================================================================
    # HUMAN-IN-THE-LOOP
    # ==========================================================================
    pending_human_review: bool = False
    human_feedback: Optional[str] = None
    human_approved: bool = False
    human_review_reason: Optional[str] = Field(
        default=None,
        description="Reason for requesting human review"
    )
    
    # ==========================================================================
    # ERROR TRACKING
    # ==========================================================================
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    # ==========================================================================
    # CONSOLIDATED OUTPUT (NEW - for frontend)
    # ==========================================================================
    consolidated_result: Optional[Dict] = Field(
        default=None,
        description="Consolidated result from all agents with scoring table and color coding"
    )
    
    class Config:
        arbitrary_types_allowed = True
