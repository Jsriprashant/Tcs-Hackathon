"""Shared state definitions for M&A Due Diligence agents."""

from typing import Annotated, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class CompanyInfo(BaseModel):
    """Company information for due diligence."""
    company_id: str
    company_name: str
    industry: str
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    employee_count: Optional[int] = None
    annual_revenue: Optional[float] = None
    
    
class RiskScore(BaseModel):
    """Risk assessment score with details."""
    category: str  # financial, legal, hr, market, overall
    score: float = Field(ge=0.0, le=1.0)  # 0 = low risk, 1 = high risk
    confidence: float = Field(ge=0.0, le=1.0)
    factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    

class AnalysisResult(BaseModel):
    """Result from an agent's analysis."""
    agent_name: str
    analysis_type: str
    summary: str
    findings: list[dict[str, Any]] = Field(default_factory=list)
    risk_score: Optional[RiskScore] = None
    supporting_data: dict[str, Any] = Field(default_factory=dict)
    confidence_level: float = Field(ge=0.0, le=1.0, default=0.8)
    timestamp: datetime = Field(default_factory=datetime.now)


class FinancialMetrics(BaseModel):
    """Key financial metrics for analysis."""
    company_id: str
    fiscal_year: int
    revenue: float
    net_income: float
    total_assets: float
    total_liabilities: float
    operating_cash_flow: float
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None


class LegalFinding(BaseModel):
    """Legal due diligence finding."""
    finding_type: Literal["litigation", "contract", "ip", "compliance"]
    severity: Literal["low", "medium", "high", "critical"]
    title: str
    description: str
    potential_liability: Optional[float] = None
    resolution_status: Optional[str] = None


class HRMetrics(BaseModel):
    """HR and organizational metrics."""
    company_id: str
    total_employees: int
    attrition_rate: float
    avg_tenure_years: float
    key_person_dependency: list[str] = Field(default_factory=list)
    pending_disputes: int = 0
    culture_score: Optional[float] = None


class BaseAgentState(BaseModel):
    """Base state shared across all agents."""
    # Core conversation
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    
    # M&A Context
    deal_type: Literal["merger", "acquisition"] = "acquisition"
    analysis_type: Literal["horizontal", "vertical"] = "horizontal"
    
    # Companies involved
    acquirer_company: Optional[CompanyInfo] = None
    target_company: Optional[CompanyInfo] = None
    
    # Analysis results from different agents
    financial_analysis: Optional[AnalysisResult] = None
    legal_analysis: Optional[AnalysisResult] = None
    hr_analysis: Optional[AnalysisResult] = None
    market_analysis: Optional[AnalysisResult] = None
    
    # Risk scores
    risk_scores: list[RiskScore] = Field(default_factory=list)
    overall_risk_score: Optional[float] = None
    deal_recommendation: Optional[str] = None
    
    # Workflow control
    current_agent: Optional[str] = None
    next_agent: Optional[str] = None
    pending_agents: list[str] = Field(default_factory=list)
    completed_agents: list[str] = Field(default_factory=list)
    
    # Human-in-the-loop
    requires_human_review: bool = False
    human_feedback: Optional[str] = None
    
    # Error handling
    errors: list[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
