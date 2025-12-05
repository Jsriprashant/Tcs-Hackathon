"""Analyst Agent tools for strategic M&A analysis using RAG retrieval."""

from typing import Any, Optional
from langchain_core.tools import tool

from src.rag_agent.tools import get_vectorstore, COLLECTIONS, normalize_company_id
from src.common.utils import format_currency, calculate_risk_score
from src.common.logging_config import get_logger

logger = get_logger(__name__)


@tool
def analyze_target_company(
    company_id: str,
) -> str:
    """
    Perform comprehensive analysis of a target company for M&A due diligence.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Comprehensive company analysis across all domains
    """
    try:
        normalized_id = normalize_company_id(company_id)
        
        results = []
        
        # Gather data from all collections
        for category, collection_name in COLLECTIONS.items():
            if category == "all":
                continue
            
            try:
                vectorstore = get_vectorstore(collection_name)
                docs = vectorstore.similarity_search(
                    f"{normalized_id}",
                    k=5,
                    filter={"company_id": normalized_id}
                )
                
                if docs:
                    results.append(f"### {category.upper()} Documents ({len(docs)} found)")
                    for doc in docs[:3]:
                        results.append(f"- {doc.metadata.get('filename', 'Unknown')}: {doc.page_content[:200]}...")
            except Exception as e:
                logger.warning(f"Error searching {collection_name}: {e}")
        
        analysis = f"""
## Target Company Analysis: {normalized_id}

### Retrieved Information:
{chr(10).join(results) if results else "Limited data available for this company."}

### Analysis Framework:

#### 1. Business Overview
- Company profile and history
- Core business activities
- Market position and competitive landscape

#### 2. Financial Health
- Revenue and profitability trends
- Balance sheet strength
- Cash flow characteristics

#### 3. Legal & Compliance
- Contract portfolio
- Litigation exposure
- Regulatory compliance status

#### 4. Human Capital
- Workforce composition
- Key personnel and dependencies
- Culture and engagement

### Key Due Diligence Questions:
1. What are the primary value drivers?
2. What are the main risk factors?
3. What synergy opportunities exist?
4. What integration challenges are expected?
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing target company: {e}")
        return f"Error analyzing target company: {str(e)}"


@tool
def estimate_synergies(
    company_id: str,
) -> str:
    """
    Estimate potential synergies from acquiring a target company.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Synergy estimation with detailed breakdown
    """
    try:
        normalized_id = normalize_company_id(company_id)
        
        # Get financial and operational data
        financial_vs = get_vectorstore(COLLECTIONS["financial"])
        hr_vs = get_vectorstore(COLLECTIONS["hr"])
        
        fin_docs = financial_vs.similarity_search(
            f"{normalized_id} revenue cost expenses operations",
            k=5,
            filter={"company_id": normalized_id}
        )
        
        hr_docs = hr_vs.similarity_search(
            f"{normalized_id} employee headcount department",
            k=5,
            filter={"company_id": normalized_id}
        )
        
        fin_content = "\n".join([doc.page_content[:300] for doc in fin_docs[:3]]) if fin_docs else "No financial data available."
        hr_content = "\n".join([doc.page_content[:300] for doc in hr_docs[:3]]) if hr_docs else "No HR data available."
        
        analysis = f"""
## Synergy Estimation: {normalized_id}

### Retrieved Financial Data:
{fin_content}

### Retrieved HR Data:
{hr_content}

### Synergy Categories:

#### 1. Cost Synergies (Year 1-2)
**Typical Sources:**
| Source | Typical Range | Confidence |
|--------|---------------|------------|
| Headcount Optimization | 3-7% of combined G&A | High |
| Facility Consolidation | 10-20% of real estate costs | Medium |
| Procurement Leverage | 2-5% of combined spend | Medium |
| IT Systems Consolidation | 5-15% of IT costs | Low-Medium |

#### 2. Revenue Synergies (Year 2-3)
**Typical Sources:**
| Source | Typical Range | Confidence |
|--------|---------------|------------|
| Cross-Selling | 1-3% of combined revenue | Low-Medium |
| Market Expansion | 2-5% of target revenue | Low |
| Pricing Power | 0.5-2% margin improvement | Medium |

#### 3. Strategic Synergies
- Technology and IP consolidation
- Customer relationship leverage
- Competitive positioning improvement

### Synergy Realization Timeline:
- Year 1: 30% of cost synergies achieved
- Year 2: 70% of cost synergies, 30% of revenue synergies
- Year 3: Full run-rate achieved

### Risk Factors:
- Integration execution risk
- Customer/employee attrition
- Market changes during integration
- Cultural alignment challenges
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error estimating synergies: {e}")
        return f"Error estimating synergies: {str(e)}"


@tool
def consolidate_due_diligence(
    company_id: str,
) -> str:
    """
    Consolidate all due diligence findings into a comprehensive assessment.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Consolidated due diligence summary with overall recommendation
    """
    try:
        normalized_id = normalize_company_id(company_id)
        
        # Gather all available data
        all_docs = []
        for category, collection_name in COLLECTIONS.items():
            if category == "all":
                continue
            
            try:
                vectorstore = get_vectorstore(collection_name)
                docs = vectorstore.similarity_search(
                    f"{normalized_id} risk assessment analysis",
                    k=5,
                    filter={"company_id": normalized_id}
                )
                for doc in docs:
                    doc.metadata["category"] = category
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Error searching {collection_name}: {e}")
        
        # Categorize findings
        findings = {
            "financial": [],
            "legal": [],
            "hr": []
        }
        
        for doc in all_docs:
            cat = doc.metadata.get("category", "unknown")
            if cat in findings:
                findings[cat].append(doc.metadata.get("filename", "Unknown"))
        
        assessment = f"""
## CONSOLIDATED DUE DILIGENCE REPORT: {normalized_id}

### Executive Summary
This report consolidates findings from financial, legal, and HR due diligence workstreams.

### Documents Analyzed:
- **Financial:** {len(findings['financial'])} documents
- **Legal:** {len(findings['legal'])} documents  
- **HR:** {len(findings['hr'])} documents

### Risk Assessment Framework:

| Category | Weight | Assessment Areas |
|----------|--------|------------------|
| Financial | 30% | Revenue quality, profitability, working capital, debt |
| Legal | 25% | Litigation, contracts, IP, compliance |
| HR/People | 15% | Attrition, key persons, culture, compliance |
| Market/Strategic | 15% | Market position, competition, growth prospects |
| Integration | 15% | Complexity, timeline, execution risk |

### Risk Level Interpretation:
- **0.0 - 0.3**: LOW RISK ✅ - Proceed with standard terms
- **0.3 - 0.5**: MODERATE RISK ℹ️ - Proceed with enhanced protections
- **0.5 - 0.7**: HIGH RISK ⚠️ - Significant price adjustment or structure changes needed
- **0.7 - 1.0**: CRITICAL RISK ⚠️ - Consider walking away

### Key Findings by Category:

#### Financial Findings
- Review revenue concentration and sustainability
- Assess working capital requirements
- Evaluate debt structure and covenants

#### Legal Findings
- Identify material litigation exposure
- Review change of control provisions
- Verify IP ownership and freedom to operate

#### HR Findings
- Assess key person dependencies
- Review attrition patterns
- Evaluate cultural compatibility

### Recommended Next Steps:
1. Complete detailed review in high-risk areas
2. Develop integration plan
3. Structure deal terms to address identified risks
4. Prepare negotiation strategy
"""
        return assessment
        
    except Exception as e:
        logger.error(f"Error consolidating due diligence: {e}")
        return f"Error consolidating due diligence: {str(e)}"


@tool
def generate_deal_recommendation(
    company_id: str,
    base_valuation: float = 0.0,
    risk_adjustment: float = 0.0,
) -> str:
    """
    Generate final deal recommendation based on due diligence findings.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
        base_valuation: Optional base valuation amount
        risk_adjustment: Optional risk adjustment factor (0-1, where 1 = 100% discount)
    
    Returns:
        Deal recommendation with pricing guidance and terms
    """
    try:
        normalized_id = normalize_company_id(company_id)
        
        # Get overview of available data
        doc_counts = {}
        for category, collection_name in COLLECTIONS.items():
            if category == "all":
                continue
            try:
                vectorstore = get_vectorstore(collection_name)
                docs = vectorstore.similarity_search(
                    normalized_id,
                    k=20,
                    filter={"company_id": normalized_id}
                )
                doc_counts[category] = len(docs)
            except:
                doc_counts[category] = 0
        
        total_docs = sum(doc_counts.values())
        
        # Calculate adjusted valuation if provided
        valuation_section = ""
        if base_valuation > 0:
            risk_adjusted = base_valuation * (1 - risk_adjustment)
            min_price = risk_adjusted * 0.9
            max_price = risk_adjusted * 1.1
            valuation_section = f"""
### Valuation Analysis
| Component | Value |
|-----------|-------|
| Base Valuation | {format_currency(base_valuation)} |
| Risk Adjustment | -{risk_adjustment*100:.0f}% |
| Risk-Adjusted Value | {format_currency(risk_adjusted)} |
| Recommended Range | {format_currency(min_price)} - {format_currency(max_price)} |
"""
        
        recommendation = f"""
## DEAL RECOMMENDATION: {normalized_id}

### Due Diligence Coverage
| Category | Documents Reviewed |
|----------|-------------------|
| Financial | {doc_counts.get('financial', 0)} |
| Legal | {doc_counts.get('legal', 0)} |
| HR | {doc_counts.get('hr', 0)} |
| **Total** | **{total_docs}** |

{valuation_section}

### Deal Structure Recommendations

#### 1. Transaction Type
- **Recommended**: Stock acquisition / Asset purchase (based on tax and liability considerations)
- **Payment Mix**: 60-70% cash, 30-40% stock or earnout

#### 2. Key Deal Terms
| Term | Recommendation |
|------|----------------|
| Escrow | 10-15% of purchase price for 18-24 months |
| Earnout | Consider if valuation gap exists |
| Non-Compete | 2-3 years for key personnel |
| Working Capital | Target as of closing with true-up |

#### 3. Representations & Warranties
- Standard R&W package with appropriate survival periods
- Specific indemnities for identified risks
- Consider R&W insurance for larger deals

#### 4. Key Conditions Precedent
- Regulatory approvals (if required)
- Third-party consents for material contracts
- No material adverse change
- Key employee retention agreements

### Timeline Estimate
| Phase | Duration |
|-------|----------|
| Signing to Close | 60-90 days |
| Regulatory (if needed) | 30-60 days |
| Integration | 12-24 months |

### Final Recommendation
Based on the available due diligence information, carefully evaluate:
1. Strategic fit and synergy potential
2. Risk-adjusted valuation
3. Integration complexity
4. Cultural alignment

**Proceed with appropriate deal protections and price adjustments to reflect identified risks.**
"""
        return recommendation
        
    except Exception as e:
        logger.error(f"Error generating deal recommendation: {e}")
        return f"Error generating deal recommendation: {str(e)}"


@tool
def compare_acquisition_targets(
    company_ids: list[str],
) -> str:
    """
    Compare multiple acquisition targets side by side.
    
    Args:
        company_ids: List of company identifiers to compare
    
    Returns:
        Comparative analysis of targets
    """
    try:
        comparison = "## Acquisition Target Comparison\n\n"
        
        target_data = {}
        
        for company_id in company_ids:
            normalized_id = normalize_company_id(company_id)
            target_data[normalized_id] = {
                "financial_docs": 0,
                "legal_docs": 0,
                "hr_docs": 0,
                "sample_content": []
            }
            
            for category, collection_name in COLLECTIONS.items():
                if category == "all":
                    continue
                
                try:
                    vectorstore = get_vectorstore(collection_name)
                    docs = vectorstore.similarity_search(
                        normalized_id,
                        k=3,
                        filter={"company_id": normalized_id}
                    )
                    
                    target_data[normalized_id][f"{category}_docs"] = len(docs)
                    if docs:
                        target_data[normalized_id]["sample_content"].append(
                            f"**{category}**: {docs[0].page_content[:150]}..."
                        )
                except Exception as e:
                    logger.warning(f"Error searching {collection_name} for {normalized_id}: {e}")
        
        # Build comparison table
        comparison += "### Data Availability\n\n"
        comparison += "| Company | Financial | Legal | HR | Total |\n"
        comparison += "|---------|-----------|-------|-----|-------|\n"
        
        for comp_id, data in target_data.items():
            total = data["financial_docs"] + data["legal_docs"] + data["hr_docs"]
            comparison += f"| {comp_id} | {data['financial_docs']} | {data['legal_docs']} | {data['hr_docs']} | {total} |\n"
        
        comparison += "\n### Sample Data by Company\n\n"
        for comp_id, data in target_data.items():
            comparison += f"#### {comp_id}\n"
            for content in data["sample_content"]:
                comparison += f"{content}\n"
            comparison += "\n"
        
        comparison += """
### Comparison Framework

When comparing targets, evaluate:

1. **Strategic Fit**
   - Alignment with growth strategy
   - Market position enhancement
   - Capability gaps filled

2. **Financial Attractiveness**
   - Revenue quality and growth
   - Margin profile
   - Capital requirements

3. **Risk Profile**
   - Legal and regulatory exposure
   - Integration complexity
   - Cultural compatibility

4. **Synergy Potential**
   - Cost reduction opportunities
   - Revenue enhancement potential
   - Strategic value creation

5. **Execution Feasibility**
   - Deal timeline
   - Financing requirements
   - Integration capability
"""
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing targets: {e}")
        return f"Error comparing targets: {str(e)}"


# Export all analyst tools
analyst_tools = [
    analyze_target_company,
    estimate_synergies,
    consolidate_due_diligence,
    generate_deal_recommendation,
    compare_acquisition_targets,
]
