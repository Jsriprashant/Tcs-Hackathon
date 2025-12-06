"""State definitions for HR Agent - M&A Policy Comparison Focus."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from src.common.state import BaseAgentState, RiskScore


# ============================================================================
# NEW: Policy Comparison Models
# ============================================================================

class PolicyParameter(BaseModel):
    """Individual policy parameter comparison between acquirer and target."""
    parameter_name: str = Field(description="Name of the parameter (e.g., 'working_hours_compensation')")
    category: str = Field(description="Category this parameter belongs to")
    weight: float = Field(description="Weight/importance of this parameter (0-100)")
    
    # Acquirer (TCS) baseline
    acquirer_baseline: dict = Field(default_factory=dict, description="TCS baseline values for this parameter")
    
    # Target company values
    target_value: Optional[dict] = Field(default=None, description="Target company's policy values")
    target_extracted_text: Optional[str] = Field(default=None, description="Relevant text extracted from target policies")
    
    # Scoring
    score: int = Field(ge=0, le=5, default=0, description="Score 0-5 for this parameter")
    max_score: int = Field(default=5, description="Maximum possible score")
    weighted_score: float = Field(default=0.0, description="Score weighted by parameter importance")
    
    # Gap analysis
    gap_description: Optional[str] = Field(default=None, description="Description of the gap between acquirer and target")
    red_flags: list[str] = Field(default_factory=list, description="Red flags identified for this parameter")
    meets_baseline: bool = Field(default=False, description="Whether target meets acquirer baseline")
    
    # Recommendations
    recommendation: Optional[str] = Field(default=None, description="Recommendation to align target with acquirer")


class PolicyCategory(BaseModel):
    """Category-level policy comparison and scoring."""
    category_name: str = Field(description="Name of policy category")
    weight: float = Field(description="Weight of this category in overall score")
    max_points: float = Field(description="Maximum points for this category")
    
    # Parameters in this category
    parameters: list[PolicyParameter] = Field(default_factory=list)
    
    # Category scoring
    earned_points: float = Field(default=0.0, description="Points earned in this category")
    score_percentage: float = Field(default=0.0, description="Percentage score for this category")
    
    # Category-level gaps
    category_gaps: list[str] = Field(default_factory=list)
    category_red_flags: list[str] = Field(default_factory=list)


class HRPolicyComparison(BaseModel):
    """Complete HR policy comparison result between acquirer and target."""
    
    # Companies
    acquirer_company: str = Field(default="TCS", description="Acquiring company name")
    target_company: str = Field(description="Target company name")
    
    # Policy analysis
    categories: list[PolicyCategory] = Field(default_factory=list)
    
    # Overall scoring
    total_score: float = Field(default=0.0, description="Overall score 0-100")
    max_score: float = Field(default=100.0, description="Maximum possible score")
    score_percentage: float = Field(default=0.0, description="Score as percentage")
    
    # Risk assessment
    risk_level: Literal["low", "medium", "high", "critical"] = Field(default="high")
    compatibility_rating: str = Field(default="Unknown", description="Compatibility rating description")
    recommendation: Literal["PROCEED", "PROCEED WITH CAUTION", "CONDITIONAL", "REJECT OR RESTRUCTURE"] = Field(default="CONDITIONAL")
    
    # Gaps and issues
    policy_gaps: list[str] = Field(default_factory=list, description="All policy gaps identified")
    all_red_flags: list[str] = Field(default_factory=list, description="All red flags identified")
    deal_breakers: list[str] = Field(default_factory=list, description="Critical issues that may block deal")
    
    # Integration recommendations
    integration_recommendations: list[str] = Field(default_factory=list)
    estimated_integration_effort: Literal["low", "medium", "high", "very_high"] = Field(default="high")


class AcquirerBaseline(BaseModel):
    """Acquirer's (TCS) HR policy baseline standards."""
    company_name: str = Field(default="TCS")
    policy_version: str = Field(default="2.0")
    effective_date: str = Field(default="December 2025")
    
    # Baseline standards loaded from knowledge base
    baseline_loaded: bool = Field(default=False)
    baseline_content: Optional[str] = Field(default=None)
    parameters: dict = Field(default_factory=dict, description="All baseline parameters")


# ============================================================================
# LEGACY: Keep for backward compatibility (Optional)
# ============================================================================

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


# ============================================================================
# UPDATED: HR Agent State for Policy Comparison
# ============================================================================

class HRAgentState(BaseAgentState):
    """State for HR Agent - Focus on Policy Comparison for M&A."""
    
    # NEW: Policy Comparison (Primary Focus)
    acquirer_baseline: Optional[AcquirerBaseline] = None
    policy_comparison: Optional[HRPolicyComparison] = None
    
    # Target company policies retrieved
    target_policies_retrieved: bool = Field(default=False)
    target_policy_documents: list[str] = Field(default_factory=list)
    
    # Legacy fields (kept for backward compatibility)
    employee_metrics: Optional[EmployeeMetrics] = None
    attrition_analysis: Optional[AttritionAnalysis] = None
    key_persons: list[KeyPersonRisk] = Field(default_factory=list)
    key_person_risk_score: float = 0.0
    hr_compliance: Optional[HRCompliance] = None
    hr_disputes_exposure: float = 0.0
    culture_assessment: Optional[CultureAssessment] = None
    integration_risk: float = 0.0
    
    # HR policies analyzed
    policies_reviewed: list[str] = Field(default_factory=list)
    policy_gaps: list[str] = Field(default_factory=list)
    
    # Overall HR risk
    hr_risk_score: Optional[RiskScore] = None
    hr_red_flags: list[str] = Field(default_factory=list)
