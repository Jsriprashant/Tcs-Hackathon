"""State definitions for Supervisor Agent."""

from typing import Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from src.common.state import BaseAgentState, RiskScore, AnalysisResult, CompanyInfo
from src.common.intent_classifier import IntentType


class SupervisorState(BaseAgentState):
    """
    State for the Supervisor Agent that orchestrates all sub-agents.
    
    The supervisor maintains the overall state of the due diligence process
    and coordinates between specialized agents.
    """
    
    # Conversation
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    
    # ==========================================================================
    # Intent Classification (NEW - Phase 2)
    # These fields gate whether the agent chain should be activated
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
        description="Whether the full agent chain should run. Only True for MA_DUE_DILIGENCE with company names"
    )
    # ==========================================================================
    
    # Deal Context
    deal_id: Optional[str] = None
    deal_type: Literal["merger", "acquisition"] = "acquisition"
    analysis_type: Literal["horizontal", "vertical"] = "horizontal"
    
    # Companies
    acquirer: Optional[CompanyInfo] = None
    target: Optional[CompanyInfo] = None
    
    # Agent routing
    next_agent: Optional[Literal[
        "finance_agent",
        "legal_agent", 
        "hr_agent",
        "analyst_agent",
        "rag_agent",
        "human",
        "FINISH"
    ]] = None
    
    # Agent results
    finance_result: Optional[AnalysisResult] = None
    legal_result: Optional[AnalysisResult] = None
    hr_result: Optional[AnalysisResult] = None
    analyst_result: Optional[AnalysisResult] = None
    
    # Risk scores from each agent
    risk_scores: dict[str, RiskScore] = Field(default_factory=dict)
    
    # Overall assessment
    overall_risk_score: Optional[float] = None
    deal_recommendation: Optional[Literal["go", "no_go", "conditional"]] = None
    recommendation_rationale: Optional[str] = None
    
    # Workflow tracking
    agents_invoked: list[str] = Field(default_factory=list)
    current_phase: Literal[
        "initialization",
        "document_retrieval", 
        "financial_analysis",
        "legal_analysis",
        "hr_analysis",
        "strategic_analysis",
        "consolidation",
        "complete"
    ] = "initialization"
    
    # Human-in-the-loop
    pending_human_review: bool = False
    human_feedback: Optional[str] = None
    human_approved: bool = False
    
    # Error tracking
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
