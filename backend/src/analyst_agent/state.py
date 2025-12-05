"""State definitions for Analyst Agent."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from src.common.state import BaseAgentState, RiskScore


class MarketPosition(BaseModel):
    """Company's market position analysis."""
    market_share: float
    market_size: float
    growth_rate: float
    competitive_rank: int
    market_trend: Literal["growing", "stable", "declining"]


class CompetitorAnalysis(BaseModel):
    """Competitor analysis results."""
    competitor_name: str
    market_share: float
    strengths: list[str]
    weaknesses: list[str]
    threat_level: Literal["low", "medium", "high"]


class SynergyEstimate(BaseModel):
    """Synergy estimation from the deal."""
    synergy_type: Literal["revenue", "cost", "operational", "technology"]
    description: str
    estimated_value: float
    confidence: float
    timeline_months: int


class DealStructure(BaseModel):
    """Recommended deal structure."""
    deal_type: Literal["stock", "cash", "mixed", "earnout"]
    recommended_price_range: tuple[float, float]
    earnout_percentage: Optional[float] = None
    key_conditions: list[str] = Field(default_factory=list)


class AnalystAgentState(BaseAgentState):
    """State for Analyst Agent operations."""
    
    # Market analysis
    market_position: Optional[MarketPosition] = None
    competitors: list[CompetitorAnalysis] = Field(default_factory=list)
    market_risk_score: float = 0.0
    
    # Strategic analysis
    synergies: list[SynergyEstimate] = Field(default_factory=list)
    total_synergy_value: float = 0.0
    integration_complexity: Literal["low", "medium", "high"] = "medium"
    
    # Deal recommendation
    deal_structure: Optional[DealStructure] = None
    deal_success_probability: float = 0.0
    
    # Sentiment analysis
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    analyst_sentiment: Optional[float] = None
    
    # Risk consolidation
    consolidated_risk_scores: dict[str, float] = Field(default_factory=dict)
    overall_recommendation: Optional[str] = None
    go_no_go_decision: Optional[Literal["go", "no_go", "conditional"]] = None
