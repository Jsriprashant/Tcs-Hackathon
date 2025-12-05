"""Utility functions for M&A Due Diligence Platform."""

from typing import Any, Optional
from datetime import datetime, date
from decimal import Decimal
import json


def format_currency(
    amount: float, 
    currency: str = "USD", 
    decimals: int = 2
) -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Numeric amount
        currency: Currency code (USD, INR, EUR)
        decimals: Decimal places
    
    Returns:
        Formatted currency string
    """
    symbols = {
        "USD": "$",
        "INR": "₹",
        "EUR": "€",
        "GBP": "£",
    }
    symbol = symbols.get(currency, currency + " ")
    
    if abs(amount) >= 1_000_000_000:
        return f"{symbol}{amount/1_000_000_000:,.{decimals}f}B"
    elif abs(amount) >= 1_000_000:
        return f"{symbol}{amount/1_000_000:,.{decimals}f}M"
    elif abs(amount) >= 1_000:
        return f"{symbol}{amount/1_000:,.{decimals}f}K"
    else:
        return f"{symbol}{amount:,.{decimals}f}"


def calculate_percentage_change(
    old_value: float, 
    new_value: float
) -> Optional[float]:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Previous value
        new_value: Current value
    
    Returns:
        Percentage change or None if old_value is zero
    """
    if old_value == 0:
        return None
    return ((new_value - old_value) / abs(old_value)) * 100


def parse_date(date_str: str) -> Optional[date]:
    """
    Parse date from various formats.
    
    Args:
        date_str: Date string in various formats
    
    Returns:
        Parsed date object or None
    """
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%d %B %Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def calculate_financial_ratios(
    revenue: float,
    net_income: float,
    total_assets: float,
    total_liabilities: float,
    current_assets: float,
    current_liabilities: float,
    total_equity: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """
    Calculate common financial ratios.
    
    Args:
        Various financial statement values
    
    Returns:
        Dictionary of calculated ratios
    """
    equity = total_equity or (total_assets - total_liabilities)
    
    return {
        "profit_margin": (net_income / revenue * 100) if revenue else None,
        "roa": (net_income / total_assets * 100) if total_assets else None,
        "roe": (net_income / equity * 100) if equity else None,
        "current_ratio": (current_assets / current_liabilities) if current_liabilities else None,
        "debt_to_equity": (total_liabilities / equity) if equity else None,
        "debt_to_assets": (total_liabilities / total_assets) if total_assets else None,
        "asset_turnover": (revenue / total_assets) if total_assets else None,
    }


def calculate_risk_score(
    factors: list[tuple[str, float, float]]
) -> tuple[float, list[str]]:
    """
    Calculate weighted risk score from multiple factors.
    
    Args:
        factors: List of (factor_name, score, weight) tuples
    
    Returns:
        Tuple of (overall_score, contributing_factors)
    """
    if not factors:
        return 0.0, []
    
    total_weight = sum(weight for _, _, weight in factors)
    if total_weight == 0:
        return 0.0, []
    
    weighted_sum = sum(score * weight for _, score, weight in factors)
    overall_score = weighted_sum / total_weight
    
    # Identify high-risk factors (score > 0.6)
    high_risk_factors = [
        f"{name}: {score:.2f}" 
        for name, score, _ in factors 
        if score > 0.6
    ]
    
    return overall_score, high_risk_factors


def merge_analysis_results(
    results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Merge multiple analysis results into a consolidated view.
    
    Args:
        results: List of analysis result dictionaries
    
    Returns:
        Merged analysis result
    """
    merged = {
        "sources": [],
        "findings": [],
        "risk_factors": [],
        "recommendations": [],
        "confidence": 0.0,
    }
    
    for result in results:
        if "source" in result:
            merged["sources"].append(result["source"])
        if "findings" in result:
            merged["findings"].extend(result["findings"])
        if "risk_factors" in result:
            merged["risk_factors"].extend(result["risk_factors"])
        if "recommendations" in result:
            merged["recommendations"].extend(result["recommendations"])
        if "confidence" in result:
            merged["confidence"] = max(merged["confidence"], result["confidence"])
    
    # Deduplicate
    merged["findings"] = list(set(merged["findings"]))
    merged["risk_factors"] = list(set(merged["risk_factors"]))
    merged["recommendations"] = list(set(merged["recommendations"]))
    
    return merged


def json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for complex objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely serialize data to JSON string."""
    return json.dumps(data, default=json_serializer, **kwargs)


def invoke_llm_with_retry(
    llm,
    messages: list,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    fallback_message: str = "The AI service is temporarily unavailable. Please try again."
) -> tuple[Any, bool]:
    """
    Invoke LLM with retry logic for transient errors.
    
    Args:
        llm: The LLM instance to invoke
        messages: Messages to send to the LLM
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (exponential backoff)
        fallback_message: Message to return if all retries fail
    
    Returns:
        Tuple of (response, success_flag)
    """
    import time
    from langchain_core.messages import AIMessage
    
    retry_delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response, True
        except Exception as e:
            error_str = str(e)
            is_transient = any(err in error_str for err in [
                "Internal Server Error", "500", "Connection", 
                "Timeout", "502", "503", "504"
            ])
            
            if is_transient:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return AIMessage(content=fallback_message), False
            else:
                raise
    
    return AIMessage(content=fallback_message), False
