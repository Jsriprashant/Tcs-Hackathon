# filepath: c:\Users\GenAIBLRANCUSR25.01HW2562306\Desktop\application_v1\Tcs-Hackathon\backend\src\legal_agent\state.py
"""State definitions for Legal Agent."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from src.common.state import BaseAgentState, RiskScore


class LitigationRecord(BaseModel):
    """A litigation or legal proceeding record."""
    case_id: str
    case_type: Literal["civil", "criminal", "regulatory", "arbitration"]
    status: Literal["pending", "ongoing", "settled", "dismissed", "closed"]
    plaintiff: str
    defendant: str
    amount_claimed: Optional[float] = None
    amount_settled: Optional[float] = None
    filing_date: Optional[str] = None
    description: str
    risk_assessment: Optional[str] = None


class ContractRisk(BaseModel):
    """Risk identified in a contract."""
    contract_type: str
    counterparty: str
    risk_type: Literal[
        "change_of_control",
        "termination_clause",
        "liability",
        "ip_assignment",
        "non_compete",
        "indemnification",
        "data_privacy"
    ]
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    potential_impact: Optional[float] = None


class IPAsset(BaseModel):
    """Intellectual property asset."""
    ip_type: Literal["patent", "trademark", "copyright", "trade_secret"]
    title: str
    status: Literal["pending", "granted", "expired", "disputed"]
    jurisdiction: str
    expiry_date: Optional[str] = None
    valuation: Optional[float] = None


class ComplianceIssue(BaseModel):
    """Compliance or regulatory issue."""
    regulation_type: str
    issue_description: str
    severity: Literal["minor", "moderate", "major", "critical"]
    remediation_status: Literal["not_started", "in_progress", "resolved"]
    potential_penalty: Optional[float] = None


class LegalAgentState(BaseAgentState):
    """State for Legal Agent operations."""
    
    # Litigation analysis
    litigations: list[LitigationRecord] = Field(default_factory=list)
    total_litigation_exposure: float = 0.0
    
    # Contract analysis
    contract_risks: list[ContractRisk] = Field(default_factory=list)
    change_of_control_issues: list[str] = Field(default_factory=list)
    
    # IP analysis
    ip_assets: list[IPAsset] = Field(default_factory=list)
    ip_valuation: Optional[float] = None
    ip_risks: list[str] = Field(default_factory=list)
    
    # Compliance
    compliance_issues: list[ComplianceIssue] = Field(default_factory=list)
    
    # Overall legal risk
    legal_risk_score: Optional[RiskScore] = None
    legal_red_flags: list[str] = Field(default_factory=list)
