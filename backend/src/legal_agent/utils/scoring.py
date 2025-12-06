"""Scoring utilities for Legal Agent MVP.

This module provides deterministic scoring calculations based on findings.
No fallbacks - scores are calculated directly from finding severities.
"""

from typing import List, Dict
from src.legal_agent.state import Finding, CategoryScore


# =============================================================================
# CONSTANTS
# =============================================================================

# Severity to point deduction mapping
SEVERITY_DEDUCTIONS: Dict[str, int] = {
    "critical": 15,
    "high": 8,
    "medium": 4,
    "low": 2,
}

# Category maximum points
CATEGORY_MAX_POINTS: Dict[str, int] = {
    "litigation": 35,
    "contracts": 35,
    "ip": 30,
}

# Risk level thresholds (score >= threshold means that level)
RISK_THRESHOLDS: List[tuple] = [
    (85, "LOW"),
    (70, "MODERATE"),
    (50, "HIGH"),
    (0, "CRITICAL"),
]


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def calculate_category_score(
    category: str,
    findings: List[Finding],
) -> CategoryScore:
    """
    Calculate score for a single category based on findings.
    
    Args:
        category: Category name (litigation, contracts, ip)
        findings: List of all findings (will filter by category)
    
    Returns:
        CategoryScore with deductions applied
    
    Example:
        >>> findings = [Finding(category="litigation", severity="critical", ...)]
        >>> score = calculate_category_score("litigation", findings)
        >>> score.points_earned  # 35 - 15 = 20
    """
    max_points = CATEGORY_MAX_POINTS[category]
    deductions = []
    total_deduction = 0
    
    # Process only findings for this category
    for finding in findings:
        if finding.category == category:
            pts = SEVERITY_DEDUCTIONS.get(finding.severity, 2)
            deductions.append({
                "finding_title": finding.title,
                "severity": finding.severity,
                "points_deducted": pts,
            })
            total_deduction += pts
    
    # Score cannot go below 0
    points_earned = max(0, max_points - total_deduction)
    
    return CategoryScore(
        category=category,
        max_points=max_points,
        points_earned=points_earned,
        deductions=deductions,
    )


def calculate_total_score(category_scores: Dict[str, CategoryScore]) -> int:
    """
    Sum all category scores to get total.
    
    Args:
        category_scores: Dict mapping category name to CategoryScore
    
    Returns:
        Total score (0-100)
    """
    return sum(score.points_earned for score in category_scores.values())


def determine_risk_level(total_score: int) -> str:
    """
    Determine risk level from total score.
    
    Args:
        total_score: Score from 0-100
    
    Returns:
        Risk level string (LOW, MODERATE, HIGH, CRITICAL)
    
    Thresholds:
        - 85-100: LOW
        - 70-84: MODERATE
        - 50-69: HIGH
        - 0-49: CRITICAL
    """
    for threshold, level in RISK_THRESHOLDS:
        if total_score >= threshold:
            return level
    return "CRITICAL"


def identify_deal_breakers(findings: List[Finding]) -> List[str]:
    """
    Identify critical findings that are deal breakers.
    
    Args:
        findings: All findings from analysis
    
    Returns:
        List of deal breaker titles (findings with severity="critical")
    """
    return [f.title for f in findings if f.severity == "critical"]


def get_deduction_for_severity(severity: str) -> int:
    """
    Get point deduction for a severity level.
    
    Args:
        severity: Severity level (critical, high, medium, low)
    
    Returns:
        Points to deduct
    """
    return SEVERITY_DEDUCTIONS.get(severity, 2)
