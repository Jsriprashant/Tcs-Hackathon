"""
MCP Server for M&A Due Diligence Orchestrator.

Provides visualization and reporting tools:
- Generate financial charts
- Create risk dashboards
- Export due diligence reports
"""

import json
from typing import Any
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("MA Due Diligence Tools")


@mcp.tool()
def generate_financial_chart(
    company_id: str,
    chart_type: str,
    metrics: list[str],
    years: list[int] = None,
) -> dict[str, Any]:
    """Generate a financial chart for visualization.
    
    Args:
        company_id: Company identifier
        chart_type: Type of chart (line, bar, pie, area)
        metrics: List of metrics to chart (revenue, net_income, ebitda, etc.)
        years: Years to include (default: last 5 years)
        
    Returns:
        Chart configuration for frontend rendering
    """
    if years is None:
        current_year = datetime.now().year
        years = list(range(current_year - 4, current_year + 1))
    
    # Generate sample data (in production, fetch from data layer)
    chart_data = {
        "type": chart_type,
        "title": f"{company_id} - Financial Metrics",
        "company_id": company_id,
        "labels": [str(y) for y in years],
        "datasets": []
    }
    
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    
    for i, metric in enumerate(metrics):
        # Sample data - would be fetched from actual data store
        base_value = 100_000_000
        growth = 1.1
        values = [base_value * (growth ** j) for j in range(len(years))]
        
        chart_data["datasets"].append({
            "label": metric.replace("_", " ").title(),
            "data": values,
            "color": colors[i % len(colors)],
        })
    
    return {
        "success": True,
        "chart": chart_data,
        "render_type": "chart",
    }


@mcp.tool()
def generate_risk_dashboard(
    company_id: str,
    include_categories: list[str] = None,
) -> dict[str, Any]:
    """Generate a risk dashboard showing all risk categories.
    
    Args:
        company_id: Company identifier
        include_categories: Risk categories to include (financial, legal, hr, strategic)
        
    Returns:
        Dashboard configuration with risk scores and indicators
    """
    if include_categories is None:
        include_categories = ["financial", "legal", "hr", "strategic"]
    
    # Sample risk scores (in production, fetch from analysis results)
    risk_scores = {
        "financial": {
            "score": 0.35,
            "level": "low",
            "indicators": [
                {"name": "Profitability", "score": 0.25, "trend": "improving"},
                {"name": "Liquidity", "score": 0.30, "trend": "stable"},
                {"name": "Solvency", "score": 0.45, "trend": "stable"},
                {"name": "Cash Flow", "score": 0.40, "trend": "improving"},
            ]
        },
        "legal": {
            "score": 0.45,
            "level": "medium",
            "indicators": [
                {"name": "Litigation", "score": 0.55, "trend": "stable"},
                {"name": "Contracts", "score": 0.35, "trend": "improving"},
                {"name": "IP Portfolio", "score": 0.40, "trend": "stable"},
                {"name": "Compliance", "score": 0.50, "trend": "declining"},
            ]
        },
        "hr": {
            "score": 0.30,
            "level": "low",
            "indicators": [
                {"name": "Attrition", "score": 0.25, "trend": "improving"},
                {"name": "Key Persons", "score": 0.40, "trend": "stable"},
                {"name": "Culture", "score": 0.30, "trend": "improving"},
                {"name": "Policies", "score": 0.25, "trend": "stable"},
            ]
        },
        "strategic": {
            "score": 0.40,
            "level": "medium",
            "indicators": [
                {"name": "Market Position", "score": 0.35, "trend": "improving"},
                {"name": "Synergy Potential", "score": 0.30, "trend": "stable"},
                {"name": "Integration Risk", "score": 0.55, "trend": "stable"},
            ]
        },
    }
    
    # Filter to requested categories
    filtered_scores = {k: v for k, v in risk_scores.items() if k in include_categories}
    
    # Calculate overall score
    weights = {"financial": 0.35, "legal": 0.30, "hr": 0.15, "strategic": 0.20}
    overall_score = sum(
        filtered_scores[cat]["score"] * weights.get(cat, 0.25)
        for cat in filtered_scores
    )
    
    return {
        "success": True,
        "dashboard": {
            "company_id": company_id,
            "overall_risk": {
                "score": round(overall_score, 2),
                "level": "low" if overall_score < 0.3 else "medium" if overall_score < 0.6 else "high",
            },
            "categories": filtered_scores,
            "generated_at": datetime.now().isoformat(),
        },
        "render_type": "dashboard",
    }


@mcp.tool()
def generate_comparison_chart(
    companies: list[str],
    metrics: list[str],
    chart_type: str = "bar",
) -> dict[str, Any]:
    """Generate a comparison chart for multiple companies.
    
    Args:
        companies: List of company identifiers to compare
        metrics: Metrics to compare
        chart_type: Type of chart (bar, radar, grouped_bar)
        
    Returns:
        Comparison chart configuration
    """
    chart_data = {
        "type": chart_type,
        "title": "Company Comparison",
        "labels": [m.replace("_", " ").title() for m in metrics],
        "datasets": []
    }
    
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    
    for i, company in enumerate(companies):
        # Sample data - would be fetched from actual data
        values = [0.5 + (hash(f"{company}_{m}") % 50) / 100 for m in metrics]
        
        chart_data["datasets"].append({
            "label": company,
            "data": values,
            "color": colors[i % len(colors)],
        })
    
    return {
        "success": True,
        "chart": chart_data,
        "render_type": "chart",
    }


@mcp.tool()
def export_due_diligence_report(
    company_id: str,
    format: str = "pdf",
    sections: list[str] = None,
) -> dict[str, Any]:
    """Export a due diligence report.
    
    Args:
        company_id: Company identifier
        format: Export format (pdf, docx, html, json)
        sections: Sections to include (executive_summary, financial, legal, hr, recommendation)
        
    Returns:
        Report generation status and download URL
    """
    if sections is None:
        sections = ["executive_summary", "financial", "legal", "hr", "recommendation"]
    
    # In production, this would generate actual report
    report_id = f"DD-{company_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "success": True,
        "report": {
            "id": report_id,
            "company_id": company_id,
            "format": format,
            "sections": sections,
            "status": "generated",
            "download_url": f"/api/reports/{report_id}.{format}",
            "generated_at": datetime.now().isoformat(),
        },
        "message": f"Due diligence report generated successfully for {company_id}",
    }


@mcp.tool()
def generate_synergy_waterfall(
    acquirer_id: str,
    target_id: str,
) -> dict[str, Any]:
    """Generate a synergy waterfall chart showing value creation.
    
    Args:
        acquirer_id: Acquirer company identifier
        target_id: Target company identifier
        
    Returns:
        Waterfall chart configuration showing synergy breakdown
    """
    # Sample synergy data (in production, from analyst agent)
    synergies = {
        "revenue_synergies": {
            "cross_selling": 15_000_000,
            "market_expansion": 25_000_000,
            "pricing_power": 8_000_000,
        },
        "cost_synergies": {
            "headcount_optimization": 12_000_000,
            "procurement_savings": 18_000_000,
            "facility_consolidation": 7_000_000,
            "technology_rationalization": 10_000_000,
        },
        "one_time_costs": {
            "integration_costs": -20_000_000,
            "severance": -8_000_000,
            "system_migration": -5_000_000,
        }
    }
    
    total_revenue = sum(synergies["revenue_synergies"].values())
    total_cost = sum(synergies["cost_synergies"].values())
    total_one_time = sum(synergies["one_time_costs"].values())
    net_synergies = total_revenue + total_cost + total_one_time
    
    waterfall_data = {
        "type": "waterfall",
        "title": f"Synergy Analysis: {acquirer_id} + {target_id}",
        "steps": [
            {"label": "Revenue Synergies", "value": total_revenue, "type": "positive"},
            {"label": "Cost Synergies", "value": total_cost, "type": "positive"},
            {"label": "One-Time Costs", "value": total_one_time, "type": "negative"},
            {"label": "Net Synergies", "value": net_synergies, "type": "total"},
        ],
        "breakdown": synergies,
    }
    
    return {
        "success": True,
        "chart": waterfall_data,
        "summary": {
            "net_synergies": net_synergies,
            "annual_run_rate": total_revenue + total_cost,
            "payback_years": abs(total_one_time) / (total_revenue + total_cost) if (total_revenue + total_cost) > 0 else 0,
        },
        "render_type": "chart",
    }


@mcp.tool()
def get_deal_scorecard(
    company_id: str,
) -> dict[str, Any]:
    """Get a comprehensive deal scorecard for a target company.
    
    Args:
        company_id: Target company identifier
        
    Returns:
        Scorecard with all dimensions and overall recommendation
    """
    # Sample scorecard (in production, aggregated from all agents)
    scorecard = {
        "company_id": company_id,
        "dimensions": {
            "financial_health": {
                "score": 7.5,
                "max_score": 10,
                "weight": 0.25,
                "highlights": [
                    "Strong revenue growth (15% CAGR)",
                    "Healthy EBITDA margins (22%)",
                    "Moderate debt levels",
                ],
                "concerns": [
                    "Working capital needs improvement",
                ]
            },
            "market_position": {
                "score": 8.0,
                "max_score": 10,
                "weight": 0.20,
                "highlights": [
                    "Top 3 market position",
                    "Strong brand recognition",
                ],
                "concerns": [
                    "Increasing competition",
                ]
            },
            "legal_risk": {
                "score": 6.5,
                "max_score": 10,
                "weight": 0.20,
                "highlights": [
                    "Clean regulatory record",
                    "Strong IP portfolio",
                ],
                "concerns": [
                    "2 pending litigation matters",
                    "Some contracts have change of control clauses",
                ]
            },
            "organizational_readiness": {
                "score": 7.0,
                "max_score": 10,
                "weight": 0.15,
                "highlights": [
                    "Low attrition rate (8%)",
                    "Strong leadership team",
                ],
                "concerns": [
                    "Key person dependency on CTO",
                ]
            },
            "synergy_potential": {
                "score": 8.5,
                "max_score": 10,
                "weight": 0.20,
                "highlights": [
                    "Significant cost synergy opportunities",
                    "Complementary product portfolio",
                ],
                "concerns": [
                    "Technology stack differences",
                ]
            },
        },
        "overall": {
            "weighted_score": 7.5,
            "recommendation": "PROCEED WITH CAUTION",
            "confidence": 0.78,
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    # Calculate weighted score
    total_score = sum(
        d["score"] * d["weight"]
        for d in scorecard["dimensions"].values()
    )
    scorecard["overall"]["weighted_score"] = round(total_score, 1)
    
    # Determine recommendation
    if total_score >= 8.0:
        scorecard["overall"]["recommendation"] = "STRONGLY RECOMMEND"
    elif total_score >= 7.0:
        scorecard["overall"]["recommendation"] = "RECOMMEND WITH CONDITIONS"
    elif total_score >= 6.0:
        scorecard["overall"]["recommendation"] = "PROCEED WITH CAUTION"
    else:
        scorecard["overall"]["recommendation"] = "NOT RECOMMENDED"
    
    return {
        "success": True,
        "scorecard": scorecard,
        "render_type": "scorecard",
    }


# Main entry point for MCP server
if __name__ == "__main__":
    mcp.run()
