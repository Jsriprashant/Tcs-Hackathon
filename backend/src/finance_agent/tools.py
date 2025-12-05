"""Finance Agent tools for financial analysis using RAG retrieval."""

from typing import Any, Optional
from langchain_core.tools import tool

from src.rag_agent.tools import retrieve_financial_documents, get_vectorstore, COLLECTIONS
from src.common.logging_config import get_logger

logger = get_logger(__name__)


@tool
def analyze_balance_sheet(
    company_id: str,
) -> str:
    """
    Analyze balance sheet data for a company including assets, liabilities, and equity.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA)
    
    Returns:
        Balance sheet analysis with key metrics and risk assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["financial"])
        docs = vectorstore.similarity_search(
            f"{company_id} balance sheet total assets liabilities equity",
            k=5,
            filter={"company_id": company_id.upper()}
        )
        
        if not docs:
            # Try without filter
            docs = vectorstore.similarity_search(
                f"{company_id} balance sheet total assets liabilities equity",
                k=5
            )
        
        if not docs:
            return f"No balance sheet data found for {company_id}"
        
        # Extract and analyze data
        content = "\n".join([doc.page_content for doc in docs])
        
        analysis = f"""
## Balance Sheet Analysis: {company_id.upper()}

### Retrieved Data:
{content}

### Key Observations:
- Review Total Assets trend over years
- Analyze debt levels (Total Liabilities)
- Check equity position and changes
- Working Capital = Current Assets - Current Liabilities

### Risk Indicators to Watch:
- High debt-to-equity ratio (>2.0)
- Negative working capital
- Declining equity
- Increasing intangible assets
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing balance sheet: {e}")
        return f"Error analyzing balance sheet: {str(e)}"


@tool
def analyze_income_statement(
    company_id: str,
) -> str:
    """
    Analyze income statement data for a company including revenue, expenses, and profitability.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA)
    
    Returns:
        Income statement analysis with profitability metrics
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["financial"])
        docs = vectorstore.similarity_search(
            f"{company_id} income statement revenue profit expenses",
            k=5,
            filter={"company_id": company_id.upper()}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{company_id} income revenue profit net income",
                k=5
            )
        
        if not docs:
            return f"No income statement data found for {company_id}"
        
        content = "\n".join([doc.page_content for doc in docs])
        
        analysis = f"""
## Income Statement Analysis: {company_id.upper()}

### Retrieved Data:
{content}

### Key Metrics to Calculate:
- Gross Margin = (Revenue - COGS) / Revenue
- Operating Margin = Operating Income / Revenue
- Net Profit Margin = Net Income / Revenue

### Profitability Assessment:
- Track revenue growth year-over-year
- Monitor expense control
- Analyze profit trends

### Red Flags:
- Declining revenue
- Shrinking margins
- Increasing operating expenses faster than revenue
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing income statement: {e}")
        return f"Error analyzing income statement: {str(e)}"


@tool
def analyze_cash_flow(
    company_id: str,
) -> str:
    """
    Analyze cash flow statement for a company including operating, investing, and financing activities.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA)
    
    Returns:
        Cash flow analysis with liquidity assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["financial"])
        docs = vectorstore.similarity_search(
            f"{company_id} cash flow operating investing financing",
            k=5,
            filter={"company_id": company_id.upper()}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{company_id} cashflow cash operations",
                k=5
            )
        
        if not docs:
            return f"No cash flow data found for {company_id}"
        
        content = "\n".join([doc.page_content for doc in docs])
        
        analysis = f"""
## Cash Flow Analysis: {company_id.upper()}

### Retrieved Data:
{content}

### Key Areas:
1. **Operating Cash Flow**: Cash from core business operations
2. **Investing Cash Flow**: Capital expenditures and investments
3. **Financing Cash Flow**: Debt and equity transactions

### Liquidity Assessment:
- Positive operating cash flow indicates healthy operations
- Free Cash Flow = Operating CF - Capital Expenditures
- Cash burn rate analysis for sustainability

### Red Flags:
- Negative operating cash flow
- Heavy reliance on financing to fund operations
- Declining cash reserves
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing cash flow: {e}")
        return f"Error analyzing cash flow: {str(e)}"


@tool
def analyze_financial_ratios(
    company_id: str,
) -> str:
    """
    Calculate and analyze key financial ratios for due diligence.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA)
    
    Returns:
        Comprehensive ratio analysis with benchmarks
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["financial"])
        
        # Get all financial data
        docs = vectorstore.similarity_search(
            f"{company_id} assets liabilities equity revenue income debt",
            k=10,
            filter={"company_id": company_id.upper()}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{company_id} financial statement",
                k=10
            )
        
        if not docs:
            return f"No financial data found for {company_id}"
        
        content = "\n".join([doc.page_content for doc in docs])
        
        analysis = f"""
## Financial Ratio Analysis: {company_id.upper()}

### Retrieved Financial Data:
{content}

### Key Ratios to Calculate:

**Liquidity Ratios:**
- Current Ratio = Current Assets / Current Liabilities (Target: >1.5)
- Quick Ratio = (Current Assets - Inventory) / Current Liabilities (Target: >1.0)

**Leverage Ratios:**
- Debt-to-Equity = Total Debt / Total Equity (Target: <2.0)
- Debt-to-Assets = Total Debt / Total Assets (Target: <0.5)

**Profitability Ratios:**
- ROE = Net Income / Shareholders' Equity (Target: >15%)
- ROA = Net Income / Total Assets (Target: >5%)

**Efficiency Ratios:**
- Asset Turnover = Revenue / Total Assets
- Working Capital Ratio = Current Assets / Current Liabilities

### Risk Assessment:
Based on the ratios, assess overall financial health and identify concerns.
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing financial ratios: {e}")
        return f"Error analyzing ratios: {str(e)}"


@tool
def assess_financial_risk(
    company_id: str,
) -> str:
    """
    Perform comprehensive financial risk assessment for M&A due diligence.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA)
    
    Returns:
        Financial risk score and detailed assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["financial"])
        
        # Retrieve all financial documents
        docs = vectorstore.similarity_search(
            f"{company_id} financial risk debt liabilities working capital",
            k=10,
            filter={"company_id": company_id.upper()}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{company_id} balance sheet income cash",
                k=10
            )
        
        content = "\n".join([doc.page_content for doc in docs[:5]])
        
        risk_assessment = f"""
## Financial Risk Assessment: {company_id.upper()}

### Data Retrieved:
{content}

### Risk Categories:

**1. Liquidity Risk**
- Assess ability to meet short-term obligations
- Working capital adequacy
- Cash runway analysis

**2. Solvency Risk**
- Long-term debt sustainability
- Interest coverage ratio
- Debt maturity schedule

**3. Profitability Risk**
- Margin sustainability
- Revenue concentration
- Cost structure flexibility

**4. Valuation Risk**
- Asset quality and potential write-downs
- Goodwill and intangibles assessment
- Off-balance sheet liabilities

### Overall Risk Score Framework:
- LOW RISK: Strong financials, positive trends
- MEDIUM RISK: Some concerns but manageable
- HIGH RISK: Significant financial concerns
- CRITICAL RISK: Major red flags, deal breaker potential

### Recommendation:
Analyze the retrieved data to determine specific risk level and key concerns.
"""
        return risk_assessment
        
    except Exception as e:
        logger.error(f"Error assessing financial risk: {e}")
        return f"Error in risk assessment: {str(e)}"


@tool
def compare_financial_performance(
    company_ids: list[str],
) -> str:
    """
    Compare financial performance across multiple companies.
    
    Args:
        company_ids: List of company identifiers to compare
    
    Returns:
        Comparative financial analysis
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["financial"])
        
        comparison = "## Financial Comparison Analysis\n\n"
        
        for company_id in company_ids:
            docs = vectorstore.similarity_search(
                f"{company_id} revenue assets equity income",
                k=3,
                filter={"company_id": company_id.upper()}
            )
            
            if docs:
                comparison += f"### {company_id.upper()}\n"
                for doc in docs:
                    comparison += f"{doc.page_content[:500]}...\n\n"
            else:
                comparison += f"### {company_id.upper()}\nNo data found.\n\n"
        
        comparison += """
### Comparison Framework:
- Revenue size and growth rates
- Profitability margins
- Asset efficiency
- Debt levels and leverage
- Cash flow generation
"""
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing financials: {e}")
        return f"Error in comparison: {str(e)}"


# Export all finance tools
finance_tools = [
    analyze_balance_sheet,
    analyze_income_statement,
    analyze_cash_flow,
    analyze_financial_ratios,
    assess_financial_risk,
    compare_financial_performance,
]
