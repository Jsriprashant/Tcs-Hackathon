# filepath: c:\Users\GenAIBLRANCUSR25.01HW2562306\Desktop\application_v1\Tcs-Hackathon\backend\src\finance_agent\state.py
"""State definitions for Finance Agent."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from src.common.state import BaseAgentState, RiskScore, FinancialMetrics


class FinancialRedFlag(BaseModel):
    """A financial red flag identified during analysis."""
    flag_type: Literal[
        "revenue_decline",
        "high_debt",
        "cash_flow_negative",
        "profit_manipulation",
        "asset_overstatement",
        "liability_understatement",
        "audit_concern",
        "going_concern"
    ]
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    fiscal_year: Optional[int] = None
    amount_involved: Optional[float] = None
    recommendation: str


class FinancialAnalysisRequest(BaseModel):
    """Request for financial analysis."""
    company_id: str
    analysis_types: list[str] = Field(
        default_factory=lambda: [
            "profitability",
            "liquidity",
            "solvency",
            "growth",
            "cash_flow",
        ]
    )
    benchmark_company_id: Optional[str] = None
    years_to_analyze: int = Field(default=5)


class FinanceAgentState(BaseAgentState):
    """State for Finance Agent operations."""
    
    # Analysis request
    analysis_request: Optional[FinancialAnalysisRequest] = None
    
    # Retrieved financial data
    financial_metrics: list[FinancialMetrics] = Field(default_factory=list)
    
    # Analysis results
    profitability_analysis: Optional[dict] = None
    liquidity_analysis: Optional[dict] = None
    solvency_analysis: Optional[dict] = None
    growth_analysis: Optional[dict] = None
    cash_flow_analysis: Optional[dict] = None
    
    # Red flags and risks
    red_flags: list[FinancialRedFlag] = Field(default_factory=list)
    financial_risk_score: Optional[RiskScore] = None
    
    # Valuation
    estimated_valuation: Optional[float] = None
    valuation_method: Optional[str] = None
    
    # Comparison with benchmarks
    benchmark_comparison: Optional[dict] = None
