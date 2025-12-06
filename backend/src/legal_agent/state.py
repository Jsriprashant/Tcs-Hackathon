"""State definitions for Legal Agent MVP.

Simplified state models for 3-category legal due diligence:
- Litigation Exposure (35 points)
- Contract Risk (35 points)
- IP Portfolio (30 points)
"""

from typing import Optional, Literal, List, Dict, Any
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


# =============================================================================
# FINDING MODEL
# =============================================================================

class Finding(BaseModel):
    """A single legal finding/red flag identified during analysis."""
    
    category: Literal["litigation", "contracts", "ip"]
    severity: Literal["critical", "high", "medium", "low"]
    title: str
    description: str
    potential_liability: Optional[float] = None
    source_document: str
    recommendation: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "litigation",
                "severity": "high",
                "title": "Pending SEC Investigation",
                "description": "Company under SEC investigation for disclosure violations",
                "potential_liability": 5000000,
                "source_document": "litigation_docket.md",
                "recommendation": "Obtain legal opinion and consider escrow provisions"
            }
        }


# =============================================================================
# CATEGORY SCORE MODEL
# =============================================================================

class CategoryScore(BaseModel):
    """Score breakdown for one analysis category."""
    
    category: Literal["litigation", "contracts", "ip"]
    max_points: int  # 35, 35, or 30
    points_earned: int
    deductions: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "litigation",
                "max_points": 35,
                "points_earned": 20,
                "deductions": [
                    {"finding_title": "SEC Investigation", "severity": "critical", "points_deducted": 15}
                ]
            }
        }


# =============================================================================
# LEGAL RESULT MODEL
# =============================================================================

class LegalResult(BaseModel):
    """Final output from legal agent analysis."""
    
    company_id: str
    company_name: str
    total_score: int  # 0-100
    max_score: int = 100
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    category_scores: Dict[str, CategoryScore]
    findings: List[Finding]
    deal_breakers: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_id": "BBD",
                "company_name": "BBD Software Com Ltd",
                "total_score": 45,
                "max_score": 100,
                "risk_level": "HIGH",
                "category_scores": {},
                "findings": [],
                "deal_breakers": ["Ongoing SEC investigation"],
                "confidence": 0.85
            }
        }


# =============================================================================
# LEGAL AGENT STATE
# =============================================================================

class LegalAgentState(BaseModel):
    """
    Simplified state for MVP Legal Agent.
    
    This state tracks the analysis progress through 3 categories:
    init → litigation → contracts → ip → scoring → complete
    """
    
    # Messages (for LangGraph compatibility)
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    
    # Company context
    company_id: str = ""
    company_name: str = ""
    
    # Phase tracking
    current_phase: Literal[
        "init",
        "litigation",
        "contracts",
        "ip",
        "scoring",
        "complete"
    ] = "init"
    
    # Collected findings from all categories
    findings: List[Finding] = Field(default_factory=list)
    
    # Category scores
    category_scores: Dict[str, CategoryScore] = Field(default_factory=dict)
    
    # Final result
    result: Optional[LegalResult] = None
    
    # Error tracking
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        # Allow arbitrary types for LangGraph message handling
        arbitrary_types_allowed = True
    legal_red_flags: list[str] = Field(default_factory=list)
