# filepath: c:\Users\GenAIBLRANCUSR25.01HW2562306\Desktop\application_v1\Tcs-Hackathon\backend\src\hr_agent\tools.py
"""HR Agent tools for HR due diligence analysis using RAG retrieval."""

from typing import Any, Optional
from langchain_core.tools import tool

from src.rag_agent.tools import get_vectorstore, COLLECTIONS, normalize_company_id
from src.common.utils import format_currency, calculate_risk_score
from src.common.logging_config import get_logger

logger = get_logger(__name__)


@tool
def analyze_employee_data(
    company_id: str,
) -> str:
    """
    Analyze employee data including headcount, demographics, and workforce composition.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Employee data analysis with workforce metrics
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for employee data
        docs = vectorstore.similarity_search(
            f"{normalized_id} employee data headcount department position salary",
            k=10,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{normalized_id} employee workforce staff",
                k=10
            )
        
        if not docs:
            return f"No employee data found for {company_id}"
        
        content = "\n\n".join([f"{doc.page_content[:500]}" for doc in docs[:8]])
        
        analysis = f"""
## Employee Data Analysis: {normalized_id}

### Retrieved Employee Records:
{content}

### Workforce Metrics:

#### 1. Headcount Overview
- Total employees and trends
- Full-time vs. part-time breakdown
- Contractor/contingent workforce

#### 2. Demographics
- Department distribution
- Position/level breakdown
- Geographic distribution

#### 3. Compensation Analysis
- Salary ranges by level
- Benefits structure
- Variable compensation

#### 4. Tenure Analysis
- Average tenure by department
- New hire vs. experienced ratio
- Long-term employee retention

#### 5. Key Observations
- Review salary distribution for market competitiveness
- Identify department staffing levels
- Note any concerning patterns in workforce composition
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing employee data: {e}")
        return f"Error analyzing employee data: {str(e)}"


@tool
def analyze_attrition(
    company_id: str,
) -> str:
    """
    Analyze employee attrition and turnover patterns.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Attrition analysis with risk assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for terminated employees and attrition data
        docs = vectorstore.similarity_search(
            f"{normalized_id} terminated termination resignation voluntary attrition turnover",
            k=10,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{normalized_id} employee left departed",
                k=10
            )
        
        if not docs:
            return f"No attrition data found for {company_id}"
        
        content = "\n\n".join([f"{doc.page_content[:400]}" for doc in docs[:8]])
        
        analysis = f"""
## Attrition Analysis: {normalized_id}

### Retrieved Termination Records:
{content}

### Attrition Assessment:

#### 1. Turnover Metrics
- Overall attrition rate
- Voluntary vs. involuntary breakdown
- Industry benchmark comparison

#### 2. Departure Patterns
- Common termination reasons
- High-risk departments/roles
- Seasonal patterns

#### 3. Voluntary Departure Analysis
- Career change motivations
- Compensation-related departures
- Cultural/management issues

#### 4. Involuntary Departures
- Performance-based terminations
- Compliance/conduct issues
- Restructuring impacts

#### 5. Risk Assessment
**Risk Levels:**
- ⚠️ CRITICAL: Attrition >1.5x industry benchmark
- ⚠️ HIGH: Multiple key person departures
- ℹ️ MEDIUM: Above-average but manageable
- ✅ LOW: Within normal range

#### 6. M&A Implications
- Flight risk during integration
- Retention package requirements
- Knowledge transfer concerns
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing attrition: {e}")
        return f"Error analyzing attrition: {str(e)}"


@tool
def analyze_key_person_dependency(
    company_id: str,
) -> str:
    """
    Analyze key person dependencies and succession risks.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Key person risk analysis with succession assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for key personnel - managers, directors, etc.
        docs = vectorstore.similarity_search(
            f"{normalized_id} manager director VP executive senior lead principal",
            k=10,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{normalized_id} key person leadership management",
                k=10
            )
        
        if not docs:
            return f"No key personnel data found for {company_id}"
        
        content = "\n\n".join([f"{doc.page_content[:400]}" for doc in docs[:8]])
        
        analysis = f"""
## Key Person Dependency Analysis: {normalized_id}

### Retrieved Key Personnel Records:
{content}

### Key Person Assessment:

#### 1. Leadership Identification
- C-suite and executive team
- Department heads
- Technical/domain experts
- Customer relationship owners

#### 2. Dependency Analysis
- Critical roles with single-point-of-failure
- Knowledge concentration risks
- Client relationship dependencies

#### 3. Succession Planning
- Formal succession plans in place
- Internal succession candidates
- Development pipeline status

#### 4. Tenure & Stability
- Leadership tenure patterns
- Recent leadership changes
- Historical stability

#### 5. Risk Assessment
**Key Person Risks:**
- ⚠️ CRITICAL: <50% succession coverage
- ⚠️ HIGH: Key founders with no succession
- ℹ️ MEDIUM: Some gaps in succession
- ✅ LOW: Strong succession coverage

#### 6. M&A Recommendations
- Retention packages for critical personnel
- Earnout structures tied to key person retention
- Knowledge transfer and documentation plans
- Stay bonuses and lock-up periods
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing key person dependency: {e}")
        return f"Error analyzing key person dependency: {str(e)}"


@tool
def analyze_hr_policies(
    company_id: str,
) -> str:
    """
    Analyze HR policies including employee handbook, benefits, and workplace policies.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        HR policy analysis with gap assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for HR policies
        docs = vectorstore.similarity_search(
            f"{normalized_id} policy handbook employee benefits leave vacation",
            k=8,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                "employee handbook policy benefits workplace conduct",
                k=8
            )
        
        if not docs:
            return f"No HR policy documents found for {company_id}"
        
        content = "\n\n".join([f"**{doc.metadata.get('filename', 'Unknown')}**:\n{doc.page_content[:500]}" for doc in docs])
        
        analysis = f"""
## HR Policy Analysis: {normalized_id}

### Retrieved Policy Documents:
{content}

### Policy Assessment:

#### 1. Employee Handbook
- Code of conduct
- Work policies and procedures
- Disciplinary processes
- Grievance procedures

#### 2. Benefits Policies
- Health and insurance benefits
- Retirement plans
- Leave policies (PTO, sick, parental)
- Other perks and benefits

#### 3. Workplace Policies
- Remote/hybrid work policies
- Diversity and inclusion
- Anti-harassment policies
- Safety and security

#### 4. Compliance Policies
- Data privacy and confidentiality
- Conflict of interest
- Ethics and compliance
- Whistleblower protections

#### 5. Gap Assessment
**Critical Policies Check:**
- ✅ Employee handbook
- ✅ Anti-discrimination policy
- ✅ Data protection policy
- ✅ Health and safety
- ❓ Review for completeness

#### 6. Integration Considerations
- Policy harmonization needs
- Benefit alignment challenges
- Cultural policy differences
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing HR policies: {e}")
        return f"Error analyzing HR policies: {str(e)}"


@tool
def analyze_hr_compliance(
    company_id: str,
) -> str:
    """
    Analyze HR compliance including employment practices and regulatory adherence.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        HR compliance analysis with liability assessment
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for compliance-related content
        docs = vectorstore.similarity_search(
            f"{normalized_id} compliance discrimination harassment safety labor employment law",
            k=8,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            # Also check legal collection for employment disputes
            legal_vectorstore = get_vectorstore(COLLECTIONS["legal"])
            docs = legal_vectorstore.similarity_search(
                f"{normalized_id} employment dispute labor complaint discrimination",
                k=5
            )
        
        if not docs:
            return f"No HR compliance records found for {company_id}"
        
        content = "\n\n".join([f"**{doc.metadata.get('filename', 'Unknown')}**:\n{doc.page_content[:500]}" for doc in docs])
        
        analysis = f"""
## HR Compliance Analysis: {normalized_id}

### Retrieved Compliance Documents:
{content}

### Compliance Assessment:

#### 1. Employment Law Compliance
- Wage and hour compliance
- Classification of employees vs. contractors
- Overtime and leave requirements
- Documentation and record-keeping

#### 2. Anti-Discrimination
- EEO compliance
- Harassment prevention programs
- Diversity metrics and initiatives
- Complaint handling procedures

#### 3. Workplace Safety
- OSHA/safety compliance
- Incident reporting
- Safety training programs
- Workers' compensation claims

#### 4. Employment Disputes
- Pending claims or investigations
- Historical litigation patterns
- Settlement history
- Regulatory actions

#### 5. Compliance Risk Rating
**Risk Categories:**
- ⚠️ CRITICAL: Pending investigations, systemic issues
- ⚠️ HIGH: Pattern of violations, multiple claims
- ℹ️ MEDIUM: Isolated incidents, addressed
- ✅ LOW: Clean compliance record

#### 6. M&A Considerations
- Successor liability for violations
- Pending claim resolution
- Post-close compliance remediation
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing HR compliance: {e}")
        return f"Error analyzing HR compliance: {str(e)}"


@tool
def analyze_culture_fit(
    company_id: str,
) -> str:
    """
    Analyze cultural factors and integration compatibility.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Culture fit analysis with integration recommendations
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Search for culture-related content
        docs = vectorstore.similarity_search(
            f"{normalized_id} culture values satisfaction engagement remote hybrid work environment",
            k=8,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{normalized_id} employee engagement satisfaction",
                k=8
            )
        
        if not docs:
            return f"No culture-related documents found for {company_id}"
        
        content = "\n\n".join([f"{doc.page_content[:400]}" for doc in docs[:6]])
        
        analysis = f"""
## Culture Fit Analysis: {normalized_id}

### Retrieved Culture Data:
{content}

### Culture Assessment:

#### 1. Work Environment
- Office vs. remote/hybrid policies
- Work-life balance indicators
- Flexibility and autonomy levels

#### 2. Employee Sentiment
- Engagement survey results
- Satisfaction indicators
- Glassdoor/external ratings

#### 3. Values & Mission
- Company values alignment
- Mission clarity
- Employee buy-in

#### 4. Management Style
- Leadership approach
- Decision-making culture
- Communication patterns

#### 5. Integration Considerations
**Culture Gap Analysis:**
- Work style compatibility
- Values alignment
- Management culture fit
- Communication style match

#### 6. Integration Recommendations
- Extended integration timeline for culture gaps
- Culture champion identification
- Communication strategy for changes
- Retention bonuses for transition period
- Cultural integration workstreams
"""
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing culture fit: {e}")
        return f"Error analyzing culture fit: {str(e)}"


@tool
def generate_hr_risk_score(
    company_id: str,
) -> str:
    """
    Generate comprehensive HR risk score for due diligence.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Overall HR risk assessment with scores and recommendations
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Get all HR documents for comprehensive assessment
        docs = vectorstore.similarity_search(
            f"{normalized_id} employee risk attrition compliance culture",
            k=15,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"{normalized_id} HR employee",
                k=15
            )
        
        doc_count = len(docs)
        doc_summary = "\n".join([f"- {doc.metadata.get('filename', 'Unknown')} ({doc.metadata.get('doc_type', 'Unknown')})" for doc in docs[:10]])
        
        assessment = f"""
## HR Risk Assessment: {normalized_id}

### Documents Analyzed: {doc_count}
{doc_summary}
{'...' if doc_count > 10 else ''}

### Risk Scoring Framework:

| Category | Weight | Assessment Areas |
|----------|--------|------------------|
| Attrition | 20% | Turnover rates, departure patterns |
| Key Person | 25% | Dependencies, succession planning |
| Compliance | 25% | Employment practices, disputes |
| Culture | 20% | Integration risk, engagement |
| Policies | 10% | Policy gaps, harmonization |

### Risk Level Interpretation:
- **0.0 - 0.3**: LOW RISK ✅ - Standard integration planning
- **0.3 - 0.5**: MODERATE RISK ℹ️ - Enhanced integration support needed
- **0.5 - 0.7**: HIGH RISK ⚠️ - Retention packages and culture initiatives required
- **0.7 - 1.0**: CRITICAL RISK ⚠️ - Significant people risks may impact deal value

### Recommended Actions:
1. Conduct stay interviews with key personnel
2. Develop retention packages for critical roles
3. Plan for cultural integration workstreams
4. Address any compliance gaps pre-close
5. Create comprehensive communication plan

### Deal Considerations:
- Retention bonus pool sizing
- Earnout structures for key persons
- Integration timeline adjustments
- Change management investment
- Post-close HR harmonization plan
"""
        return assessment
        
    except Exception as e:
        logger.error(f"Error generating HR risk score: {e}")
        return f"Error generating HR risk score: {str(e)}"


# Export all HR tools
hr_tools = [
    analyze_employee_data,
    analyze_attrition,
    analyze_key_person_dependency,
    analyze_hr_policies,
    analyze_hr_compliance,
    analyze_culture_fit,
    generate_hr_risk_score,
]
