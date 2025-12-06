"""
MCP Server for Social Media Sentiment Analysis.

Provides tools to:
- Analyze sentiment of user opinions on companies
- Get company opinions from social media
- Auto-updates opinions every 30 seconds using LLM
"""

import json
import sys
import os
from typing import Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

# Import social media tools functions directly
from src.tools.social_media_tools import (
    analyze_company_sentiment,
    get_company_opinions,
    load_opinions_data,
    start_background_updater
)

# Initialize MCP server
mcp = FastMCP("Social Media Sentiment Analysis")

# Start background updater for opinions
_updater_thread = None


@mcp.tool()
def analyze_sentiment(company_name: str) -> dict[str, Any]:
    """
    Analyze sentiment of user opinions on a specific company.
    
    This tool retrieves all social media opinions about the specified company
    and performs sentiment analysis to determine overall public perception.
    
    Args:
        company_name: Name of the company to analyze (e.g., "BBD Softwares", "Supernova", "Technobox")
        
    Returns:
        Comprehensive sentiment analysis including:
        - Overall sentiment (positive/negative/mixed)
        - Average sentiment score (-1 to 1)
        - Breakdown of positive/negative/neutral opinions
        - Top positive and negative opinions
        - All analyzed opinions with individual sentiment scores
    
    Example:
        analyze_sentiment("BBD Softwares")
        analyze_sentiment("Supernova")
    """
    result = analyze_company_sentiment(company_name)
    return result


@mcp.tool()
def get_opinions(company_name: str, limit: int = 10) -> dict[str, Any]:
    """
    Get recent user opinions about a specific company.
    
    Args:
        company_name: Name of the company to get opinions for
        limit: Maximum number of opinions to return (default: 10)
        
    Returns:
        List of recent opinions with user info, platform, and engagement metrics
    """
    opinions = get_company_opinions(company_name)
    
    if not opinions:
        data = load_opinions_data()
        available = list(data.get("companies", {}).keys())
        return {
            "success": False,
            "error": f"No opinions found for '{company_name}'",
            "available_companies": available
        }
    
    # Sort by timestamp (most recent first) and limit
    sorted_opinions = sorted(
        opinions, 
        key=lambda x: x.get("timestamp", ""), 
        reverse=True
    )[:limit]
    
    return {
        "success": True,
        "company": company_name,
        "total_opinions": len(opinions),
        "returned_opinions": len(sorted_opinions),
        "opinions": sorted_opinions
    }


@mcp.tool()
def list_companies() -> dict[str, Any]:
    """
    List all companies available for sentiment analysis.
    
    Returns:
        List of companies with their descriptions and industries
    """
    data = load_opinions_data()
    companies = data.get("companies", {})
    
    company_list = []
    for name, info in companies.items():
        opinion_count = len([o for o in data.get("opinions", []) if o.get("company") == name])
        company_list.append({
            "name": name,
            "description": info.get("description", ""),
            "industry": info.get("industry", ""),
            "total_opinions": opinion_count
        })
    
    return {
        "success": True,
        "total_companies": len(company_list),
        "companies": company_list,
        "last_updated": data.get("metadata", {}).get("last_updated", "")
    }


@mcp.tool()
def get_sentiment_summary() -> dict[str, Any]:
    """
    Get a summary of sentiment across all companies.
    
    Returns:
        Overview of public sentiment for all tracked companies
    """
    data = load_opinions_data()
    companies = data.get("companies", {})
    
    summary = []
    for company_name in companies.keys():
        result = analyze_company_sentiment(company_name)
        if result.get("success") and result.get("total_opinions", 0) > 0:
            summary.append({
                "company": company_name,
                "industry": companies[company_name].get("industry", ""),
                "total_opinions": result["total_opinions"],
                "overall_sentiment": result["overall_sentiment"],
                "sentiment_score": result["average_sentiment_score"],
                "positive_percentage": result["sentiment_breakdown"]["positive_percentage"],
                "negative_percentage": result["sentiment_breakdown"]["negative_percentage"]
            })
    
    # Sort by sentiment score
    summary.sort(key=lambda x: x["sentiment_score"], reverse=True)
    
    return {
        "success": True,
        "analysis_timestamp": datetime.now().isoformat(),
        "total_companies_analyzed": len(summary),
        "company_rankings": summary,
        "best_sentiment": summary[0] if summary else None,
        "worst_sentiment": summary[-1] if summary else None
    }


@mcp.tool()
def compare_companies(company_names: list[str]) -> dict[str, Any]:
    """
    Compare sentiment between multiple companies.
    
    Args:
        company_names: List of company names to compare
        
    Returns:
        Side-by-side comparison of sentiment metrics
    """
    comparisons = []
    
    for company_name in company_names:
        result = analyze_company_sentiment(company_name)
        if result.get("success"):
            comparisons.append({
                "company": result.get("company"),
                "total_opinions": result.get("total_opinions", 0),
                "overall_sentiment": result.get("overall_sentiment", "unknown"),
                "sentiment_score": result.get("average_sentiment_score", 0),
                "sentiment_breakdown": result.get("sentiment_breakdown", {})
            })
        else:
            comparisons.append({
                "company": company_name,
                "error": result.get("error", "Analysis failed")
            })
    
    # Determine winner
    valid_comparisons = [c for c in comparisons if "error" not in c]
    winner = max(valid_comparisons, key=lambda x: x["sentiment_score"]) if valid_comparisons else None
    
    return {
        "success": True,
        "comparison": comparisons,
        "best_sentiment": winner["company"] if winner else None,
        "analysis_timestamp": datetime.now().isoformat()
    }


def main():
    """Main entry point for the MCP server."""
    global _updater_thread
    
    # Start background updater
    _updater_thread = start_background_updater()
    
    # Run MCP server
    mcp.run()


if __name__ == "__main__":
    main()