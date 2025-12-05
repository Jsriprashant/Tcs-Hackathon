"""HR Agent module for HR due diligence."""

from .graph import hr_agent, graph
from .tools import (
    analyze_employee_data,
    analyze_attrition,
    analyze_key_person_dependency,
    analyze_hr_policies,
    analyze_hr_compliance,
    analyze_culture_fit,
    generate_hr_risk_score,
    hr_tools,
)

__all__ = [
    "hr_agent",
    "graph",
    "analyze_employee_data",
    "analyze_attrition",
    "analyze_key_person_dependency",
    "analyze_hr_policies",
    "analyze_hr_compliance",
    "analyze_culture_fit",
    "generate_hr_risk_score",
    "hr_tools",
]
