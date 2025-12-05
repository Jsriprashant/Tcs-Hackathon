"""Analyst Agent module for strategic M&A analysis."""

from .graph import analyst_agent, graph
from .tools import (
    analyze_target_company,
    estimate_synergies,
    consolidate_due_diligence,
    generate_deal_recommendation,
    compare_acquisition_targets,
    analyst_tools,
)

__all__ = [
    "analyst_agent",
    "graph",
    "analyze_target_company",
    "estimate_synergies",
    "consolidate_due_diligence",
    "generate_deal_recommendation",
    "compare_acquisition_targets",
    "analyst_tools",
]
