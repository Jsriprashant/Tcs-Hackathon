"""
Supervisor Agent module for orchestrating M&A due diligence.

ENHANCED v2.0:
- Intelligent intent classification with scope detection
- Dynamic analysis planning based on query type
- Risk aggregation with weighted domain scoring
- Master analyst for final deal recommendations
- Support for single-domain and full analyses
- All responses LLM-driven (no hardcoded responses)
"""

from .graph import graph, build_supervisor_graph
from .state import SupervisorState
from .prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    ROUTING_PROMPT,
    CONSOLIDATION_PROMPT,
    MASTER_ANALYST_PROMPT,
    INTELLIGENT_ROUTING_PROMPT,
    DOMAIN_SUMMARY_PROMPT,
    RISK_AGGREGATION_PROMPT,
    GREETING_PROMPT,
    HELP_PROMPT,
    INFORMATIONAL_REDIRECT_PROMPT,
)

# Enhanced models (v2.0)
from .models import (
    AnalysisScope,
    DealType,
    EnhancedIntentResult,
    AgentOutput,
    DomainRiskScore,
    AggregatedRisk,
    DealAnalysis,
    ReasoningStep,
    AnalysisPlan,
    RiskLevel,
    Recommendation,
    DOMAIN_WEIGHTS,
    calculate_risk_level,
    get_recommendation_from_risk,
)

# Analysis planner (v2.0)
from .planner import (
    create_analysis_plan,
    get_next_agents,
    is_plan_complete,
    get_pending_agents,
    SCOPE_PLAN_TEMPLATES,
)

__all__ = [
    # Graph
    "graph",
    "build_supervisor_graph",
    
    # State
    "SupervisorState",
    
    # Prompts
    "SUPERVISOR_SYSTEM_PROMPT",
    "ROUTING_PROMPT",
    "CONSOLIDATION_PROMPT",
    "MASTER_ANALYST_PROMPT",
    "INTELLIGENT_ROUTING_PROMPT",
    "DOMAIN_SUMMARY_PROMPT",
    "RISK_AGGREGATION_PROMPT",
    "GREETING_PROMPT",
    "HELP_PROMPT",
    "INFORMATIONAL_REDIRECT_PROMPT",
    
    # Models (v2.0)
    "AnalysisScope",
    "DealType",
    "EnhancedIntentResult",
    "AgentOutput",
    "DomainRiskScore",
    "AggregatedRisk",
    "DealAnalysis",
    "ReasoningStep",
    "AnalysisPlan",
    "RiskLevel",
    "Recommendation",
    "DOMAIN_WEIGHTS",
    "calculate_risk_level",
    "get_recommendation_from_risk",
    
    # Planner (v2.0)
    "create_analysis_plan",
    "get_next_agents",
    "is_plan_complete",
    "get_pending_agents",
    "SCOPE_PLAN_TEMPLATES",
]
