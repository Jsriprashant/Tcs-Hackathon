"""Legal Agent module for legal due diligence."""

from .graph import legal_agent, graph
from .tools import (
    analyze_contracts,
    analyze_litigation_exposure,
    analyze_ip_portfolio,
    analyze_regulatory_compliance,
    analyze_corporate_governance,
    generate_legal_risk_score,
    legal_tools,
)

__all__ = [
    "legal_agent",
    "graph",
    "analyze_contracts",
    "analyze_litigation_exposure",
    "analyze_ip_portfolio",
    "analyze_regulatory_compliance",
    "analyze_corporate_governance",
    "generate_legal_risk_score",
    "legal_tools",
]
