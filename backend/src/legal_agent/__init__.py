"""Legal Agent MVP module for M&A legal due diligence.

This module provides a simplified 3-category legal analysis:
- Litigation Exposure (35 points)
- Contract Risk (35 points)  
- IP Portfolio (30 points)
"""

from .graph import legal_agent, graph, build_legal_agent_graph
from .state import LegalAgentState, Finding, CategoryScore, LegalResult
from .prompts import COMPANY_NAMES

__all__ = [
    # Graph
    "legal_agent",
    "graph",
    "build_legal_agent_graph",
    # State models
    "LegalAgentState",
    "Finding",
    "CategoryScore",
    "LegalResult",
    # Constants
    "COMPANY_NAMES",
]
