"""
Finance Agent Tools - Focused tools for TCS M&A Financial Analysis.

Design Philosophy:
- LLM handles reasoning, interpretation, and flexible data extraction
- Tools handle deterministic math calculations only
- Benchmarks loaded from external JSON file (scalable, maintainable)
- Minimal tools = less complexity, more reliable

Context Engineering:
- Lazy loading: Benchmarks loaded only when needed
- File-based config: Easy to update without code changes
"""

import json
import os
from typing import Any
from pathlib import Path
from langchain_core.tools import tool

from src.rag_agent.tools import retrieve_financial_documents as rag_retrieve
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# BENCHMARK LOADING (Lazy, Cached)
# =============================================================================

_benchmarks_cache = None

def _get_benchmarks() -> dict:
    """Load TCS benchmarks from JSON file. Cached after first load."""
    global _benchmarks_cache
    
    if _benchmarks_cache is not None:
        return _benchmarks_cache
    
    try:
        # Get path relative to this file
        current_dir = Path(__file__).parent
        benchmark_path = current_dir / "knowledge" / "tcs_benchmarks.json"
        
        with open(benchmark_path, 'r') as f:
            _benchmarks_cache = json.load(f)
        
        logger.info(f"Loaded TCS benchmarks v{_benchmarks_cache.get('version', 'unknown')}")
        return _benchmarks_cache
        
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        # Return minimal defaults if file fails
        return {"benchmarks": {}, "deal_breakers": [], "category_max_scores": {}}


# =============================================================================
# TOOL 1: GET FINANCIAL DOCUMENTS (RAG Wrapper)
# =============================================================================

@tool
def get_financial_documents(
    company_id: str,
    doc_type: str = "all"
) -> str:
    """
    Retrieve financial documents for a target company from the document database.
    
    Args:
        company_id: Company identifier (e.g., BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
        doc_type: Type of document to retrieve:
                  - "income_statement" - Revenue, expenses, profitability
                  - "balance_sheet" - Assets, liabilities, equity
                  - "cash_flow" - Operating, investing, financing cash flows
                  - "all" - All financial statements (default)
    
    Returns:
        Financial document content with key metrics for analysis
    """
    try:
        # Map doc_type to search queries
        queries = {
            "income_statement": f"{company_id} income statement revenue profit expenses net income EBITDA",
            "balance_sheet": f"{company_id} balance sheet assets liabilities equity debt",
            "cash_flow": f"{company_id} cash flow operating investing financing free cash flow",
            "all": f"{company_id} financial statements income balance sheet cash flow"
        }
        
        query = queries.get(doc_type, queries["all"])
        
        # Use RAG agent's retrieve function
        result = rag_retrieve.invoke({
            "company_id": company_id,
            "query": query,
            "k": 10 if doc_type == "all" else 5
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving financial documents: {e}")
        return f"Error retrieving documents for {company_id}: {str(e)}"


# =============================================================================
# TOOL 2: CALCULATE RATIOS (Deterministic Math)
# =============================================================================

@tool
def calculate_ratios(metrics: str) -> str:
    """
    Calculate financial ratios from extracted metrics. Use this for precise calculations.
    
    Args:
        metrics: JSON string with financial metrics. Expected keys:
                - revenue, cost_of_revenue, gross_profit
                - operating_income, ebitda, net_income, interest_expense
                - total_assets, total_liabilities, total_equity, total_debt
                - current_assets, current_liabilities, inventory
                - operating_cash_flow, capital_expenditure, free_cash_flow
                - prior_year_revenue (for YoY growth)
    
    Returns:
        JSON string with calculated ratios (percentages and multiples)
    """
    try:
        # Parse input metrics
        data = json.loads(metrics) if isinstance(metrics, str) else metrics
        
        def safe_divide(a, b):
            """Safely divide, return None if division not possible."""
            if a is None or b is None or b == 0:
                return None
            return a / b
        
        # Extract values with defaults
        revenue = data.get("revenue") or data.get("total_revenue")
        cost_of_revenue = data.get("cost_of_revenue") or data.get("cogs")
        gross_profit = data.get("gross_profit")
        operating_income = data.get("operating_income") or data.get("ebit")
        ebitda = data.get("ebitda")
        net_income = data.get("net_income")
        interest_expense = data.get("interest_expense")
        
        total_assets = data.get("total_assets")
        total_liabilities = data.get("total_liabilities")
        total_equity = data.get("total_equity")
        total_debt = data.get("total_debt")
        current_assets = data.get("current_assets")
        current_liabilities = data.get("current_liabilities")
        inventory = data.get("inventory", 0) or 0
        
        operating_cash_flow = data.get("operating_cash_flow") or data.get("ocf")
        capital_expenditure = data.get("capital_expenditure") or data.get("capex")
        free_cash_flow = data.get("free_cash_flow") or data.get("fcf")
        
        prior_revenue = data.get("prior_year_revenue")
        
        # Calculate Free Cash Flow if not provided
        if free_cash_flow is None and operating_cash_flow and capital_expenditure:
            free_cash_flow = operating_cash_flow + capital_expenditure
        
        # Calculate ratios
        ratios = {}
        
        # === PROFITABILITY RATIOS ===
        ratios["gross_profit_margin"] = safe_divide(gross_profit, revenue)
        ratios["net_profit_margin"] = safe_divide(net_income, revenue)
        ratios["operating_margin"] = safe_divide(operating_income, revenue)
        ratios["ebitda_margin"] = safe_divide(ebitda, revenue)
        
        # === RETURNS RATIOS ===
        ratios["roe"] = safe_divide(net_income, total_equity)
        ratios["roa"] = safe_divide(net_income, total_assets)
        
        if total_assets and current_liabilities:
            capital_employed = total_assets - current_liabilities
            ratios["roce"] = safe_divide(operating_income or ebitda, capital_employed)
        else:
            ratios["roce"] = None
        
        # === LIQUIDITY RATIOS ===
        ratios["current_ratio"] = safe_divide(current_assets, current_liabilities)
        
        if current_assets is not None:
            ratios["quick_ratio"] = safe_divide(current_assets - inventory, current_liabilities)
        else:
            ratios["quick_ratio"] = None
        
        if current_assets is not None and current_liabilities is not None:
            ratios["working_capital"] = current_assets - current_liabilities
        else:
            ratios["working_capital"] = None
        
        # === LEVERAGE RATIOS ===
        ratios["debt_to_equity"] = safe_divide(total_debt, total_equity)
        ratios["debt_to_assets"] = safe_divide(total_debt, total_assets)
        ratios["interest_coverage"] = safe_divide(operating_income or ebitda, interest_expense)
        
        # === EFFICIENCY RATIOS ===
        ratios["asset_turnover"] = safe_divide(revenue, total_assets)
        
        # === GROWTH RATIOS ===
        if revenue and prior_revenue and prior_revenue > 0:
            ratios["revenue_yoy_growth"] = (revenue - prior_revenue) / prior_revenue
        else:
            ratios["revenue_yoy_growth"] = None
        
        # === CASH FLOW RATIOS ===
        ratios["ocf_to_net_income"] = safe_divide(operating_cash_flow, net_income)
        ratios["fcf_margin"] = safe_divide(free_cash_flow, revenue)
        
        # Format output
        output = {"calculated_ratios": {}, "input_metrics_used": {}}
        
        for key, value in ratios.items():
            if value is not None:
                if key in ["current_ratio", "quick_ratio", "debt_to_equity", 
                          "debt_to_assets", "interest_coverage", "asset_turnover",
                          "ocf_to_net_income"]:
                    output["calculated_ratios"][key] = round(value, 2)
                elif key == "working_capital":
                    output["calculated_ratios"][key] = round(value, 0)
                else:
                    output["calculated_ratios"][key] = round(value * 100, 1)
            else:
                output["calculated_ratios"][key] = "N/A"
        
        output["input_metrics_used"] = {
            "revenue": revenue,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_equity": total_equity,
            "total_debt": total_debt
        }
        
        return json.dumps(output, indent=2)
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. Details: {e}"
    except Exception as e:
        logger.error(f"Error calculating ratios: {e}")
        return f"Error calculating ratios: {str(e)}"


# =============================================================================
# TOOL 3: CALCULATE TCS SCORE (Scoring Algorithm)
# =============================================================================

@tool  
def calculate_tcs_score(analysis_input: str) -> str:
    """
    Calculate the TCS M&A Financial Score (0-100) based on ratios and red flags.
    Benchmarks are loaded from knowledge/tcs_benchmarks.json for scalability.
    
    Args:
        analysis_input: JSON string with structure:
                       {
                           "ratios": {ratio_name: value, ...},
                           "red_flags": ["flag1", "flag2", ...]
                       }
    
    Returns:
        JSON with TCS score breakdown, interpretation, and recommendation
    """
    try:
        data = json.loads(analysis_input) if isinstance(analysis_input, str) else analysis_input
        ratios = data.get("ratios", {})
        red_flags = data.get("red_flags", [])
        
        # Load benchmarks from external file
        config = _get_benchmarks()
        benchmarks = config.get("benchmarks", {})
        deal_breaker_patterns = config.get("deal_breakers", [])
        category_max = config.get("category_max_scores", {})
        score_ranges = config.get("score_interpretation", {})
        
        # Build scoring rules from config
        SCORING_RULES = {}
        CATEGORY_MAP = {}
        
        for category, metrics in benchmarks.items():
            for metric_name, metric_config in metrics.items():
                SCORING_RULES[metric_name] = (
                    metric_config.get("target", 0),
                    metric_config.get("points", 0),
                    metric_config.get("higher_better", True)
                )
                # Map metric to category
                if category in ["profitability"]:
                    CATEGORY_MAP[metric_name] = "profitability"
                elif category in ["liquidity"]:
                    CATEGORY_MAP[metric_name] = "liquidity"
                elif category in ["leverage"]:
                    CATEGORY_MAP[metric_name] = "leverage"
                elif category in ["returns", "growth"]:
                    CATEGORY_MAP[metric_name] = "growth_returns"
                elif category in ["efficiency"]:
                    CATEGORY_MAP[metric_name] = "efficiency"
        
        category_scores = {
            "profitability": 0, "liquidity": 0, "leverage": 0,
            "growth_returns": 0, "efficiency": 0
        }
        
        metric_details = {}
        
        for metric, (benchmark, max_pts, higher_better) in SCORING_RULES.items():
            value = ratios.get(metric)
            
            if value is None or value == "N/A":
                metric_details[metric] = {"score": 0, "status": "NO_DATA"}
                continue
            
            if isinstance(value, str):
                try:
                    value = float(value.replace("%", ""))
                except:
                    continue
            
            # Scoring logic: Full points if meets benchmark, half if 50%+, zero otherwise
            if higher_better:
                if value >= benchmark:
                    score, status = max_pts, "GREEN"
                elif value >= benchmark * 0.5:
                    score, status = max_pts * 0.5, "AMBER"
                else:
                    score, status = 0, "RED"
            else:
                if value <= benchmark:
                    score, status = max_pts, "GREEN"
                elif value <= benchmark * 2:
                    score, status = max_pts * 0.5, "AMBER"
                else:
                    score, status = 0, "RED"
            
            metric_details[metric] = {
                "value": value, "benchmark": benchmark,
                "score": score, "max_score": max_pts, "status": status
            }
            
            # Add to category
            cat = CATEGORY_MAP.get(metric)
            if cat:
                category_scores[cat] += score
        
        raw_total = sum(category_scores.values())
        
        # Red flag penalties using patterns from config
        penalty = 0
        deal_breaker = False
        
        for flag in red_flags:
            flag_lower = flag.lower().replace(" ", "_").replace("-", "_")
            
            # Check against deal breaker patterns
            if any(db.replace("-", "_") in flag_lower for db in deal_breaker_patterns):
                deal_breaker = True
                break
            elif "critical" in flag_lower:
                penalty += 10
            elif "high" in flag_lower:
                penalty += 5
            else:
                penalty += 2
        
        # Determine final score and recommendation
        if deal_breaker:
            final_score = 0
            interpretation = "Very High Risk"
            recommendation = "REJECT"
            action = "❌ Deal breaker detected. Rejection recommended."
        else:
            final_score = max(0, raw_total - penalty)
            
            # Use score ranges from config
            if final_score >= 76:
                range_info = score_ranges.get("76-100", {})
                interpretation = range_info.get("level", "Strong")
                recommendation = range_info.get("action", "PROCEED")
                action = f"✅ {range_info.get('desc', 'Good candidate for acquisition.')}"
            elif final_score >= 51:
                range_info = score_ranges.get("51-75", {})
                interpretation = range_info.get("level", "Moderate")
                recommendation = range_info.get("action", "CAUTION")
                action = f"⚠️ {range_info.get('desc', 'Proceed with valuation discount.')}"
            elif final_score >= 26:
                range_info = score_ranges.get("26-50", {})
                interpretation = range_info.get("level", "Weak")
                recommendation = range_info.get("action", "CAUTION")
                action = f"⚠️ {range_info.get('desc', 'Deep forensic audit required.')}"
            else:
                range_info = score_ranges.get("0-25", {})
                interpretation = range_info.get("level", "Very High Risk")
                recommendation = range_info.get("action", "REJECT")
                action = f"❌ {range_info.get('desc', 'High financial risk.')}"
        
        result = {
            "tcs_score": {
                "total": round(final_score, 1),
                "raw_score": round(raw_total, 1),
                "penalty": round(penalty, 1),
                "interpretation": interpretation
            },
            "category_breakdown": {
                cat: {"score": score, "max": category_max.get(cat, 25)}
                for cat, score in category_scores.items()
            },
            "metric_details": metric_details,
            "recommendation": recommendation,
            "action": action,
            "deal_breaker_detected": deal_breaker,
            "config_version": config.get("version", "unknown")
        }
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. Details: {e}"
    except Exception as e:
        logger.error(f"Error calculating TCS score: {e}")
        return f"Error calculating TCS score: {str(e)}"


# =============================================================================
# TOOL 4: GET BENCHMARKS (Optional - for LLM context injection)
# =============================================================================

@tool
def get_tcs_benchmarks() -> str:
    """
    Get TCS M&A benchmarks for reference during analysis.
    Use this when you need to know the specific benchmark targets.
    
    Returns:
        Compressed benchmark summary table
    """
    config = _get_benchmarks()
    benchmarks = config.get("benchmarks", {})
    
    lines = ["TCS M&A Benchmarks:"]
    lines.append("Category|Metric|Target|RedFlag")
    lines.append("-|-|-|-")
    
    for category, metrics in benchmarks.items():
        for metric, values in metrics.items():
            target = values.get("target", "N/A")
            red_flag = values.get("red_flag", "N/A")
            symbol = ">" if values.get("higher_better", True) else "<"
            lines.append(f"{category}|{metric}|{symbol}{target}|{symbol}{red_flag}")
    
    deal_breakers = config.get("deal_breakers", [])
    lines.append(f"\nDeal Breakers: {', '.join(deal_breakers[:5])}...")
    
    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

finance_tools = [
    get_financial_documents,
    calculate_ratios,
    calculate_tcs_score,
    get_tcs_benchmarks,
]

# Export benchmark loader for use in prompts
get_benchmarks = _get_benchmarks
