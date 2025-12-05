# filepath: c:\Users\GenAIBLRANCUSR25.01HW2562306\Desktop\application_v1\Tcs-Hackathon\backend\src\hr_agent\state.py
"""State definitions for HR Agent."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from src.common.state import BaseAgentState, RiskScore


class EmployeeMetrics(BaseModel):
    """Employee-related metrics."""
    total_employees: int
    full_time: int
    part_time: int
    contractors: int
    average_tenure_years: float
    median_salary: Optional[float] = None


class AttritionAnalysis(BaseModel):
    """Employee attrition analysis."""
    annual_attrition_rate: float
    voluntary_turnover: float
    involuntary_turnover: float
    key_person_departures: int
    industry_benchmark: float


class KeyPersonRisk(BaseModel):
    """Key person dependency risk."""
    person_name: str
    role: str
    tenure_years: float
    risk_if_departed: Literal["low", "medium", "high", "critical"]
    succession_plan: bool
    retention_measures: list[str] = Field(default_factory=list)


class HRCompliance(BaseModel):
    """HR compliance status."""
    employment_disputes: int
    discrimination_claims: int
    labor_violations: int
    pending_union_issues: bool
    whistleblower_cases: int


class CultureAssessment(BaseModel):
    """Company culture assessment."""
    employee_satisfaction_score: Optional[float] = None
    glassdoor_rating: Optional[float] = None
    culture_alignment_score: Optional[float] = None
    integration_concerns: list[str] = Field(default_factory=list)


class HRAgentState(BaseAgentState):
    """State for HR Agent operations."""
    
    # Employee data
    employee_metrics: Optional[EmployeeMetrics] = None
    attrition_analysis: Optional[AttritionAnalysis] = None
    
    # Key person risk
    key_persons: list[KeyPersonRisk] = Field(default_factory=list)
    key_person_risk_score: float = 0.0
    
    # HR compliance
    hr_compliance: Optional[HRCompliance] = None
    hr_disputes_exposure: float = 0.0
    
    # Culture
    culture_assessment: Optional[CultureAssessment] = None
    integration_risk: float = 0.0
    
    # HR policies analyzed
    policies_reviewed: list[str] = Field(default_factory=list)
    policy_gaps: list[str] = Field(default_factory=list)
    
    # Overall HR risk
    hr_risk_score: Optional[RiskScore] = None
    hr_red_flags: list[str] = Field(default_factory=list)
