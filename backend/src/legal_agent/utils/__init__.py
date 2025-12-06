"""Legal Agent utility modules.

Exports scoring and retrieval functions for use by the legal agent graph.
"""

from .scoring import (
    calculate_category_score,
    calculate_total_score,
    determine_risk_level,
    identify_deal_breakers,
    get_deduction_for_severity,
    SEVERITY_DEDUCTIONS,
    CATEGORY_MAX_POINTS,
    RISK_THRESHOLDS,
)

from .retrieval import (
    retrieve_for_category,
    retrieve_company_docs,
    retrieve_benchmark_docs,
    get_normalized_company_id,
    CATEGORY_CONFIG,
    BENCHMARK_QUERIES,
)

__all__ = [
    # Scoring functions
    "calculate_category_score",
    "calculate_total_score",
    "determine_risk_level",
    "identify_deal_breakers",
    "get_deduction_for_severity",
    # Scoring constants
    "SEVERITY_DEDUCTIONS",
    "CATEGORY_MAX_POINTS",
    "RISK_THRESHOLDS",
    # Retrieval functions
    "retrieve_for_category",
    "retrieve_company_docs",
    "retrieve_benchmark_docs",
    "get_normalized_company_id",
    # Retrieval constants
    "CATEGORY_CONFIG",
    "BENCHMARK_QUERIES",
]
