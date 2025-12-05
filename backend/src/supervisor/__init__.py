"""Supervisor Agent module for orchestrating M&A due diligence."""

from .graph import graph, build_supervisor_graph
from .state import SupervisorState
from .prompts import SUPERVISOR_SYSTEM_PROMPT, ROUTING_PROMPT, CONSOLIDATION_PROMPT

__all__ = [
    "graph",
    "build_supervisor_graph",
    "SupervisorState",
    "SUPERVISOR_SYSTEM_PROMPT",
    "ROUTING_PROMPT",
    "CONSOLIDATION_PROMPT",
]
