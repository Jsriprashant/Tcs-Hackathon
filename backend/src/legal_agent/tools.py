"""Legal Agent tools for legal due diligence analysis using RAG retrieval."""

from typing import Any, Optional
from langchain_core.tools import tool

from src.rag_agent.tools import get_vectorstore, COLLECTIONS, normalize_company_id
from src.common.utils import format_currency, calculate_risk_score
from src.common.logging_config import get_logger

logger = get_logger(__name__)


@tool
def analyze_contracts(
    company_id: str,
) -> str:
    """
    Analyze contracts and agreements for a company including change of control provisions.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Contract analysis with key terms, risks, and change of control assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for contracts
        docs = vectorstore.similarity_search(
            f"{normalized_id} contract agreement terms conditions termination assignment",
            k=8,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                "contract agreement terms conditions change of control assignment termination",
                k=8
            )
        
        if not docs:
            return f"No contract documents found for {company_id}"
        
        content = "\n\n".join([f"**{doc.metadata.get('filename', 'Unknown')}**:\n{doc.page_content[:800]}" for doc in docs])
        
        analysis = f"""
## Contract Analysis: {normalized_id}

### Retrieved Contracts:
{content}

### Key Analysis Areas:

#### 1. Change of Control Provisions
- Review assignment clauses for restrictions
- Identify contracts requiring counterparty consent for M&A
- Note any automatic termination rights on ownership change

#### 2. Material Contract Terms
- Term length and renewal provisions
- Termination rights and notice periods
- Payment terms and pricing structures

#### 3. Risk Assessment
**Key Questions:**
- Are there contracts that could be terminated on acquisition?
- What consents are needed from counterparties?
- Are there exclusivity or non-compete provisions?

#### 4. Recommendations
- Obtain waivers/consents before closing
- Budget for potential contract renegotiations
- Consider deal structure to minimize CoC triggers
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing contracts: {e}")
        return f"Error analyzing contracts: {str(e)}"


@tool
def analyze_litigation_exposure(
    company_id: str,
) -> str:
    """
    Analyze litigation exposure and legal disputes for a company.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Litigation exposure analysis with case details and estimated liability
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for litigation documents
        docs = vectorstore.similarity_search(
            f"{normalized_id} litigation lawsuit court case judgment penalty dispute claim",
            k=8,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                "litigation lawsuit court case judgment penalty regulatory order complaint",
                k=8
            )
        
        if not docs:
            return f"No litigation records found for {company_id}. This may indicate a clean litigation history or data needs to be loaded."
        
        content = "\n\n".join([f"**{doc.metadata.get('filename', 'Unknown')}** ({doc.metadata.get('doc_type', 'Unknown')}):\n{doc.page_content[:600]}" for doc in docs])
        
        analysis = f"""
## Litigation Exposure Analysis: {normalized_id}

### Retrieved Litigation Records:
{content}

### Analysis Framework:

#### 1. Active Litigation
- Pending cases and claims
- Total amounts at stake
- Expected timelines for resolution

#### 2. Historical Patterns
- Previous settlements and judgments
- Pattern of recurring issues
- Compliance track record

#### 3. Liability Assessment
**Risk Categories:**
- ⚠️ CRITICAL: Class actions, regulatory enforcement
- ⚠️ HIGH: Material pending claims >$1M
- ℹ️ MEDIUM: Standard commercial disputes
- ✅ LOW: Minor claims, strong defenses

#### 4. M&A Implications
- Indemnification requirements
- Escrow provisions for pending claims
- Representations and warranties coverage

#### 5. Recommendations
- Review all pending claims in detail
- Assess potential liability for unresolved disputes
- Include appropriate reps and warranties in deal docs
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing litigation: {e}")
        return f"Error analyzing litigation: {str(e)}"


@tool
def analyze_ip_portfolio(
    company_id: str,
) -> str:
    """
    Analyze intellectual property portfolio including patents, trademarks, and copyrights.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        IP portfolio analysis with valuation and risk assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for IP documents
        docs = vectorstore.similarity_search(
            f"{normalized_id} patent trademark copyright intellectual property license IP assignment",
            k=8,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                "patent trademark copyright intellectual property license open source",
                k=8
            )
        
        if not docs:
            return f"No IP documents found for {company_id}"
        
        content = "\n\n".join([f"**{doc.metadata.get('filename', 'Unknown')}**:\n{doc.page_content[:600]}" for doc in docs])
        
        analysis = f"""
## IP Portfolio Analysis: {normalized_id}

### Retrieved IP Documents:
{content}

### Portfolio Assessment:

#### 1. Patent Portfolio
- Number and scope of patents
- Technology coverage areas
- Expiration timeline
- Freedom to operate considerations

#### 2. Trademarks
- Brand protection coverage
- Geographic scope
- Registration status

#### 3. Copyrights & Trade Secrets
- Software and content IP
- Documentation of ownership
- Employee invention assignments

#### 4. License Agreements
- Inbound licenses (dependencies)
- Outbound licenses (revenue)
- Open source compliance

#### 5. Risk Assessment
**Key Concerns:**
- ⚠️ Patents expiring within 2 years
- ⚠️ Ongoing IP disputes
- ⚠️ Freedom to operate issues
- ⚠️ Open source license compliance

#### 6. Valuation Considerations
- Core vs. non-core IP assets
- Licensing revenue potential
- Defensive value of portfolio
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing IP portfolio: {e}")
        return f"Error analyzing IP portfolio: {str(e)}"


@tool
def analyze_regulatory_compliance(
    company_id: str,
) -> str:
    """
    Analyze regulatory compliance status including licenses, permits, and violations.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Compliance assessment with risk factors and recommendations
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for compliance documents
        docs = vectorstore.similarity_search(
            f"{normalized_id} compliance regulatory license permit GDPR SOX environmental tax labor",
            k=8,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                "compliance regulatory license permit data protection environmental tax filing labor law",
                k=8
            )
        
        if not docs:
            return f"No compliance documents found for {company_id}"
        
        content = "\n\n".join([f"**{doc.metadata.get('filename', 'Unknown')}**:\n{doc.page_content[:600]}" for doc in docs])
        
        analysis = f"""
## Regulatory Compliance Analysis: {normalized_id}

### Retrieved Compliance Documents:
{content}

### Compliance Assessment:

#### 1. Data Protection & Privacy
- GDPR compliance status
- Data processing agreements
- Privacy policy adequacy
- Breach notification procedures

#### 2. Industry-Specific Regulations
- Required licenses and permits
- Industry certifications
- Sector-specific compliance

#### 3. Environmental Compliance
- Environmental permits
- Waste management compliance
- ESG considerations

#### 4. Tax & Financial Compliance
- Tax filing status
- Transfer pricing documentation
- Audit history

#### 5. Labor Law Compliance
- Employment regulations
- Health and safety requirements
- Union agreements

#### 6. Compliance Risk Rating
**Risk Categories:**
- ⚠️ CRITICAL: Pending regulatory investigations
- ⚠️ HIGH: Pattern of violations, failed audits
- ℹ️ MEDIUM: Minor gaps, remediation in progress
- ✅ LOW: Clean compliance record

#### 7. M&A Implications
- Required regulatory approvals
- Notification requirements
- Post-closing compliance integration
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing compliance: {e}")
        return f"Error analyzing compliance: {str(e)}"


@tool
def analyze_corporate_governance(
    company_id: str,
) -> str:
    """
    Analyze corporate governance documents including board structure and bylaws.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Corporate governance analysis with structural assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for governance documents
        docs = vectorstore.similarity_search(
            f"{normalized_id} corporate governance board bylaws articles incorporation shareholder",
            k=5,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                "corporate governance board resolution bylaws articles incorporation",
                k=5
            )
        
        if not docs:
            return f"No corporate governance documents found for {company_id}"
        
        content = "\n\n".join([f"**{doc.metadata.get('filename', 'Unknown')}**:\n{doc.page_content[:600]}" for doc in docs])
        
        analysis = f"""
## Corporate Governance Analysis: {normalized_id}

### Retrieved Governance Documents:
{content}

### Governance Assessment:

#### 1. Corporate Structure
- Legal entity structure
- Subsidiary relationships
- Jurisdictions of incorporation

#### 2. Board Composition
- Board members and roles
- Independence of directors
- Committee structures

#### 3. Shareholder Rights
- Voting rights and classes
- Preemptive rights
- Tag-along/drag-along provisions

#### 4. M&A Approval Requirements
- Board approval thresholds
- Shareholder vote requirements
- Anti-takeover provisions

#### 5. Risk Assessment
- Governance best practices adherence
- Potential structural impediments to deal
- Required corporate approvals
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing corporate governance: {e}")
        return f"Error analyzing corporate governance: {str(e)}"


@tool
def generate_legal_risk_score(
    company_id: str,
) -> str:
    """
    Generate comprehensive legal risk score based on all legal due diligence areas.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Overall legal risk assessment with scores and recommendations
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        # Get all legal documents for comprehensive assessment
        docs = vectorstore.similarity_search(
            f"{normalized_id} risk litigation compliance contract IP regulatory",
            k=15,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{normalized_id} legal",
                k=15
            )
        
        doc_count = len(docs)
        doc_summary = "\n".join([f"- {doc.metadata.get('filename', 'Unknown')} ({doc.metadata.get('doc_type', 'Unknown')})" for doc in docs[:10]])
        
        assessment = f"""
## Legal Risk Assessment: {normalized_id}

### Documents Analyzed: {doc_count}
{doc_summary}
{'...' if doc_count > 10 else ''}

### Risk Scoring Framework:

| Category | Weight | Assessment Areas |
|----------|--------|------------------|
| Litigation | 30% | Pending cases, historical patterns, exposure amount |
| Contracts | 25% | CoC provisions, material terms, counterparty risks |
| IP | 20% | Portfolio strength, disputes, FTO issues |
| Compliance | 25% | Regulatory status, violations, investigations |

### Risk Level Interpretation:
- **0.0 - 0.3**: LOW RISK ✅ - Standard legal provisions sufficient
- **0.3 - 0.5**: MODERATE RISK ℹ️ - Enhanced reps and warranties needed
- **0.5 - 0.7**: HIGH RISK ⚠️ - Consider escrow and indemnification
- **0.7 - 1.0**: CRITICAL RISK ⚠️ - Significant legal DD required before proceeding

### Recommended Actions:
1. Complete detailed review of all pending litigation
2. Obtain counterparty consents for material contracts
3. Verify IP ownership and conduct FTO analysis
4. Confirm regulatory compliance status
5. Structure appropriate indemnification provisions

### Deal Considerations:
- Escrow requirements for pending claims
- Special indemnities for identified risks
- Representations and warranties insurance
- Post-closing compliance obligations
"""
        return assessment
        
    except Exception as e:
        logger.error(f"Error generating legal risk score: {e}")
        return f"Error generating legal risk score: {str(e)}"


# Export all legal tools
legal_tools = [
    analyze_contracts,
    analyze_litigation_exposure,
    analyze_ip_portfolio,
    analyze_regulatory_compliance,
    analyze_corporate_governance,
    generate_legal_risk_score,
]
