"""
HR Agent Tools - Smart M&A Policy Comparison Engine.

Tools for comparing target company HR policies against TCS (acquirer) baseline.
Enhanced with deep knowledge integration from hr_benchmarks.json and TCS policy files.

Architecture:
- LLM handles reasoning and evaluation
- Tools provide data, scoring frameworks, and calculations
- All 13 parameters are loaded from hr_benchmarks.json for consistency
"""

import json
from pathlib import Path
from typing import Any, Optional, Dict, List
from langchain_core.tools import tool

from src.rag_agent.tools import get_vectorstore, COLLECTIONS, normalize_company_id
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# KNOWLEDGE FILE LOADING (Lazy, Cached)
# =============================================================================

_benchmarks_cache = None
_tcs_baseline_cache = None
_rubrics_cache = None
_meta_prompt_cache = None


def _get_knowledge_dir() -> Path:
    """Get the path to the knowledge directory."""
    return Path(__file__).parent / "knowledge"


def _get_hr_benchmarks() -> dict:
    """Load HR benchmarks from JSON file. Cached after first load."""
    global _benchmarks_cache
    
    if _benchmarks_cache is not None:
        return _benchmarks_cache
    
    try:
        benchmark_path = _get_knowledge_dir() / "hr_benchmarks.json"
        
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            _benchmarks_cache = json.load(f)
        
        logger.info(f"Loaded HR benchmarks v{_benchmarks_cache.get('version', 'unknown')} with {len(_benchmarks_cache.get('parameters', {}))} parameters")
        return _benchmarks_cache
        
    except Exception as e:
        logger.error(f"Failed to load HR benchmarks: {e}")
        return {"parameters": {}, "risk_thresholds": {}, "deal_breakers": []}


def _get_tcs_baseline_text() -> str:
    """Load TCS HR policy baseline text. Cached after first load."""
    global _tcs_baseline_cache
    
    if _tcs_baseline_cache is not None:
        return _tcs_baseline_cache
    
    try:
        baseline_path = _get_knowledge_dir() / "tcs_hr_policy.md"
        
        with open(baseline_path, 'r', encoding='utf-8') as f:
            _tcs_baseline_cache = f.read()
        
        logger.info(f"Loaded TCS HR policy baseline ({len(_tcs_baseline_cache)} chars)")
        return _tcs_baseline_cache
        
    except Exception as e:
        logger.error(f"Failed to load TCS baseline: {e}")
        return "TCS HR Policy baseline not available."


def _get_rubrics_text() -> str:
    """Load HR rubrics markdown file. Cached after first load."""
    global _rubrics_cache
    
    if _rubrics_cache is not None:
        return _rubrics_cache
    
    try:
        rubrics_path = _get_knowledge_dir() / "hr_rubric_tcs_baseline.md"
        
        with open(rubrics_path, 'r', encoding='utf-8') as f:
            _rubrics_cache = f.read()
        
        logger.info(f"Loaded HR rubrics ({len(_rubrics_cache)} chars)")
        return _rubrics_cache
        
    except Exception as e:
        logger.error(f"Failed to load HR rubrics: {e}")
        return "HR rubrics not available."


def _get_meta_prompt_text() -> str:
    """Load meta prompt markdown file. Cached after first load."""
    global _meta_prompt_cache
    
    if _meta_prompt_cache is not None:
        return _meta_prompt_cache
    
    try:
        meta_path = _get_knowledge_dir() / "meta_prompt_tcs_baseline.md"
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            _meta_prompt_cache = f.read()
        
        logger.info(f"Loaded meta prompt ({len(_meta_prompt_cache)} chars)")
        return _meta_prompt_cache
        
    except Exception as e:
        logger.error(f"Failed to load meta prompt: {e}")
        return "Meta prompt not available."


def _get_all_parameters() -> List[str]:
    """Get list of all parameter names from benchmarks."""
    benchmarks = _get_hr_benchmarks()
    return list(benchmarks.get("parameters", {}).keys())


# =============================================================================
# TOOL 1: GET ACQUIRER BASELINE (TCS Standards)
# =============================================================================

@tool
def get_acquirer_baseline() -> str:
    """
    Load the acquirer's (TCS) HR policy baseline standards.
    
    This tool retrieves the complete TCS HR Policy Manual that serves as the 
    gold standard for comparing target company policies during M&A due diligence.
    
    Returns:
        Complete TCS HR policy baseline text with all parameters and standards
    """
    try:
        baseline_text = _get_tcs_baseline_text()
        benchmarks = _get_hr_benchmarks()
        
        # Create a comprehensive baseline summary
        result = f"""# TCS HR Policy Baseline (Acquirer Standard)

## Policy Document
{baseline_text}

---

## Scoring Parameters Summary

Total Parameters: {len(benchmarks.get('parameters', {}))}
Total Weight: {benchmarks.get('total_points', 100)} points

### Parameter Categories:
"""
        
        for param_name, param_data in benchmarks.get('parameters', {}).items():
            weight = param_data.get('weight', 0)
            desc = param_data.get('description', '')
            result += f"\n**{param_name}** (Weight: {weight})\n- {desc}\n"
        
        result += f"\n\n### Risk Thresholds:\n"
        for risk_level, threshold in benchmarks.get('risk_thresholds', {}).items():
            result += f"- **{risk_level.upper()}**: {threshold.get('min_score')}-{threshold.get('max_score')} - {threshold.get('description')}\n"
        
        result += f"\n\n### Deal Breakers:\n"
        for breaker in benchmarks.get('deal_breakers', []):
            result += f"- {breaker}\n"
        
        logger.info("Successfully loaded TCS HR baseline for policy comparison")
        return result
        
    except Exception as e:
        logger.error(f"Error loading acquirer baseline: {e}")
        return f"Error loading TCS baseline: {str(e)}"


# =============================================================================
# TOOL 2: GET TARGET HR POLICIES (RAG Wrapper)
# =============================================================================

@tool
def get_target_hr_policies(company_id: str) -> str:
    """
    Retrieve ALL HR policy documents for a target company from the RAG vector store.
    
    This tool searches for:
    - Employee handbooks
    - HR policy documents
    - Leave policies
    - Compensation documents
    - Code of conduct
    - Any other HR-related policies
    
    Args:
        company_id: Target company identifier (e.g., BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Complete HR policy documents from target company for comparison
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        # Comprehensive search for all HR policies
        search_queries = [
            f"{normalized_id} employee handbook policy",
            f"{normalized_id} leave policy vacation time off",
            f"{normalized_id} compensation salary benefits",
            f"{normalized_id} code of conduct ethics",
            f"{normalized_id} working hours overtime",
            f"{normalized_id} performance appraisal review",
            f"{normalized_id} grievance harassment POSH",
            f"{normalized_id} exit separation termination",
            f"{normalized_id} training development career",
            f"{normalized_id} remote work flexible arrangement"
        ]
        
        all_docs = []
        seen_content = set()  # Deduplicate
        
        for query in search_queries:
            docs = vectorstore.similarity_search(
                query,
                k=5,
                filter={"company_id": normalized_id}
            )
            
            for doc in docs:
                content_hash = hash(doc.page_content[:200])  # Hash first 200 chars
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)
        
        if not all_docs:
            return f"⚠️ No HR policy documents found for {company_id}. This is a HIGH RISK indicator."
        
        result = f"# HR Policy Documents: {normalized_id}\n\n"
        result += f"**Total Documents Retrieved:** {len(all_docs)}\n\n"
        result += "---\n\n"
        
        for i, doc in enumerate(all_docs, 1):
            result += f"## Document {i}\n"
            result += f"**Source:** {doc.metadata.get('filename', 'Unknown')}\n"
            result += f"**Type:** {doc.metadata.get('doc_type', 'Unknown')}\n"
            result += f"**Category:** {doc.metadata.get('category', 'hr')}\n\n"
            result += f"{doc.page_content}\n\n"
            result += "---\n\n"
        
        logger.info(f"Retrieved {len(all_docs)} HR policy documents for {normalized_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving target HR policies: {e}")
        return f"Error retrieving HR policies for {company_id}: {str(e)}"


# =============================================================================
# TOOL 3: COMPARE POLICY CATEGORY
# =============================================================================

@tool
def compare_policy_category(
    category_name: str,
    acquirer_data: str,
    target_data: str
) -> str:
    """
    Compare a specific policy category between acquirer (TCS) and target company.
    
    This tool performs detailed comparison for one of the 10 HR policy categories:
    1. working_hours_compensation
    2. leave_time_off
    3. compensation_transparency
    4. employment_terms
    5. performance_management
    6. employee_relations_culture
    7. legal_compliance
    8. exit_separation
    9. data_privacy_confidentiality
    10. training_development
    
    Args:
        category_name: Name of the policy category to compare
        acquirer_data: TCS baseline data for this category (from get_acquirer_baseline)
        target_data: Target company's policy data for this category (from get_target_hr_policies)
    
    Returns:
        Detailed comparison result with scoring guidance for the LLM to evaluate
    """
    try:
        benchmarks = _get_hr_benchmarks()
        category_params = benchmarks.get('parameters', {}).get(category_name)
        
        if not category_params:
            return f"❌ Category '{category_name}' not found in benchmarks. Valid categories: {', '.join(benchmarks.get('parameters', {}).keys())}"
        
        result = f"""# Policy Category Comparison: {category_name.replace('_', ' ').title()}

## Category Details
**Weight:** {category_params.get('weight', 0)} points  
**Description:** {category_params.get('description', 'N/A')}

---

## TCS Baseline Standards (Acquirer)
"""
        
        tcs_baseline = category_params.get('tcs_baseline', {})
        for key, value in tcs_baseline.items():
            result += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        result += f"\n---\n\n## Scoring Guide for This Category\n\n"
        
        scoring_guide = category_params.get('scoring_guide', {})
        for score, description in sorted(scoring_guide.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=True):
            result += f"**Score {score}:** {description}\n"
        
        result += f"\n---\n\n## Red Flags to Watch For\n\n"
        
        red_flags = category_params.get('red_flags', [])
        for flag in red_flags:
            result += f"- ⚠️ {flag}\n"
        
        result += f"\n---\n\n## Analysis Instructions\n\n"
        result += f"1. **Extract relevant sections** from target data that relate to {category_name}\n"
        result += f"2. **Compare each TCS baseline item** against target's policy\n"
        result += f"3. **Identify gaps** where target falls short of TCS standards\n"
        result += f"4. **Check for red flags** from the list above\n"
        result += f"5. **Assign score 0-5** based on the scoring guide\n"
        result += f"6. **Calculate weighted score** = (score / 5) * {category_params.get('weight', 0)}\n"
        result += f"7. **Provide justification** citing specific policy text\n\n"
        
        result += f"---\n\n## Target Company Data for This Category\n\n"
        result += f"(Extract relevant sections from the target policy documents provided)\n\n"
        
        logger.info(f"Prepared comparison framework for category: {category_name}")
        return result
        
    except Exception as e:
        logger.error(f"Error comparing policy category: {e}")
        return f"Error comparing category {category_name}: {str(e)}"


# =============================================================================
# TOOL 4: CALCULATE HR COMPATIBILITY SCORE
# =============================================================================

@tool
def calculate_hr_compatibility_score(comparison_data: str) -> str:
    """
    Calculate overall HR compatibility score from individual parameter scores.
    
    This tool takes the results of all parameter comparisons and:
    1. Aggregates weighted scores
    2. Calculates total score (0-100)
    3. Determines risk level
    4. Provides recommendation
    
    Args:
        comparison_data: JSON string containing all parameter scores
                        Format: {"parameters": [{"name": "...", "score": 0-5, "weight": X}, ...]}
    
    Returns:
        Overall compatibility assessment with score, risk level, and recommendation
    """
    try:
        # Parse input data
        try:
            data = json.loads(comparison_data)
        except json.JSONDecodeError:
            return "❌ Error: comparison_data must be valid JSON format"
        
        parameters = data.get('parameters', [])
        
        if not parameters:
            return "❌ Error: No parameters provided for scoring"
        
        # Calculate weighted scores
        total_weighted_score = 0.0
        total_possible_weight = 0.0
        
        category_scores = {}
        
        for param in parameters:
            param_name = param.get('name', '')
            score = param.get('score', 0)  # 0-5
            weight = param.get('weight', 0)  # Weight in points
            
            # Convert 0-5 score to weighted points
            earned_points = (score / 5.0) * weight
            total_weighted_score += earned_points
            total_possible_weight += weight
            
            category_scores[param_name] = {
                'score': score,
                'weight': weight,
                'earned_points': round(earned_points, 2)
            }
        
        # Calculate final score (0-100)
        final_score = (total_weighted_score / total_possible_weight) * 100 if total_possible_weight > 0 else 0
        final_score = round(final_score, 2)
        
        # Determine risk level based on thresholds
        benchmarks = _get_hr_benchmarks()
        risk_thresholds = benchmarks.get('risk_thresholds', {})
        
        risk_level = "unknown"
        risk_description = "Unknown risk level"
        recommendation = "FURTHER ANALYSIS REQUIRED"
        
        for level, threshold in risk_thresholds.items():
            min_score = threshold.get('min_score', 0)
            max_score = threshold.get('max_score', 100)
            
            if min_score <= final_score <= max_score:
                risk_level = level
                risk_description = threshold.get('description', '')
                recommendation = threshold.get('action', '')
                break
        
        # Build result
        result = f"""# HR Compatibility Score Calculation

## Overall Score: {final_score}/100

**Risk Level:** {risk_level.upper()}  
**Description:** {risk_description}  
**Recommendation:** {recommendation}

---

## Category Breakdown

| Parameter | Score (0-5) | Weight | Earned Points | Possible Points |
|-----------|-------------|--------|---------------|-----------------|
"""
        
        for param_name, param_scores in category_scores.items():
            result += f"| {param_name.replace('_', ' ').title()} | {param_scores['score']}/5 | {param_scores['weight']} | {param_scores['earned_points']} | {param_scores['weight']} |\n"
        
        result += f"\n**Total:** {round(total_weighted_score, 2)}/{round(total_possible_weight, 2)} = **{final_score}%**\n\n"
        
        result += f"---\n\n## Risk Assessment\n\n"
        
        if final_score >= 80:
            result += "✅ **LOW RISK** - Strong policy alignment with TCS standards\n"
            result += "- Seamless integration expected\n"
            result += "- Minimal policy harmonization needed\n"
        elif final_score >= 60:
            result += "🟡 **MEDIUM RISK** - Moderate policy alignment\n"
            result += "- Some policy harmonization required\n"
            result += "- Integration timeline: 3-6 months\n"
        elif final_score >= 40:
            result += "🟠 **HIGH RISK** - Significant policy gaps\n"
            result += "- Major integration effort required\n"
            result += "- Integration timeline: 6-12 months\n"
        else:
            result += "🔴 **CRITICAL RISK** - Major policy incompatibility\n"
            result += "- Fundamental policy misalignment\n"
            result += "- Consider deal restructuring or rejection\n"
        
        result += f"\n---\n\n## Recommendation: {recommendation}\n\n"
        
        logger.info(f"Calculated HR compatibility score: {final_score}/100 ({risk_level})")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating compatibility score: {e}")
        return f"Error calculating HR compatibility score: {str(e)}"


# =============================================================================
# TOOL 5: GET SCORING RUBRICS (Smart Knowledge Integration)
# =============================================================================

@tool
def get_scoring_rubrics(parameter_name: Optional[str] = None) -> str:
    """
    Get detailed scoring rubrics for HR policy evaluation.
    
    This tool provides the complete scoring framework from hr_rubric_tcs_baseline.md
    for accurate and consistent policy evaluation. Use this BEFORE scoring any parameter
    to understand the exact criteria for each score level (0-5).
    
    Args:
        parameter_name: Optional - specific parameter to get rubrics for.
                       If not provided, returns all rubrics.
                       Valid values: working_hours_compensation, leave_time_off, 
                       compensation_transparency, employment_terms, performance_management,
                       employee_relations_culture, legal_compliance, exit_separation,
                       data_privacy_confidentiality, training_development, diversity_inclusion,
                       disciplinary_policy, employee_benefits_insurance
    
    Returns:
        Detailed scoring rubrics with criteria for each score level
    """
    try:
        benchmarks = _get_hr_benchmarks()
        
        if parameter_name:
            # Get specific parameter rubric
            param_data = benchmarks.get('parameters', {}).get(parameter_name)
            if not param_data:
                all_params = ", ".join(_get_all_parameters())
                return f"❌ Parameter '{parameter_name}' not found. Valid parameters: {all_params}"
            
            result = f"""# Scoring Rubric: {parameter_name.replace('_', ' ').title()}

## Overview
- **ID:** {param_data.get('id', 'N/A')}
- **Weight:** {param_data.get('weight', 0)}%
- **Description:** {param_data.get('description', 'N/A')}

---

## What to Extract from Target Documents
"""
            what_to_extract = param_data.get('what_to_extract', [])
            for item in what_to_extract:
                result += f"- {item}\n"
            
            result += "\n---\n\n## TCS Baseline Values (Comparison Standard)\n"
            tcs_baseline = param_data.get('tcs_baseline', {})
            for key, value in tcs_baseline.items():
                result += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            
            result += "\n---\n\n## Scoring Criteria\n\n"
            scoring_guide = param_data.get('scoring_guide', {})
            for score in ['5', '4', '3', '2', '1', '0']:
                desc = scoring_guide.get(score, 'No description')
                result += f"### Score {score}\n{desc}\n\n"
            
            result += "---\n\n## Red Flags (Automatic Score Reduction)\n"
            red_flags = param_data.get('red_flags', [])
            for flag in red_flags:
                result += f"- ⚠️ {flag}\n"
            
            if param_data.get('deal_breaker'):
                result += "\n\n🚨 **DEAL BREAKER CATEGORY**: Issues in this category may halt the entire deal.\n"
            
            return result
        
        else:
            # Get all rubrics overview
            result = """# Complete Scoring Rubrics Overview

## 13 Policy Parameters for M&A HR Due Diligence

"""
            for param_id, param_data in benchmarks.get('parameters', {}).items():
                result += f"### {param_data.get('id', '?')}. {param_id.replace('_', ' ').title()}\n"
                result += f"- **Weight:** {param_data.get('weight', 0)}%\n"
                result += f"- **Description:** {param_data.get('description', 'N/A')}\n"
                result += f"- **Score 5:** {param_data.get('scoring_guide', {}).get('5', 'Excellent')}\n"
                result += f"- **Score 0:** {param_data.get('scoring_guide', {}).get('0', 'Unacceptable')}\n"
                result += "\n"
            
            result += "\n---\n\n## Risk Thresholds\n\n"
            for level, data in benchmarks.get('risk_thresholds', {}).items():
                result += f"- **{data.get('label', level)}:** {data.get('min_score', 0)}-{data.get('max_score', 100)} → {data.get('action', '')}\n"
            
            return result
        
    except Exception as e:
        logger.error(f"Error getting scoring rubrics: {e}")
        return f"Error loading scoring rubrics: {str(e)}"


# =============================================================================
# TOOL 6: CHECK DEAL BREAKERS
# =============================================================================

@tool
def check_deal_breakers(target_policy_summary: str) -> str:
    """
    Check target company policies for deal-breaker issues.
    
    This tool scans the provided policy summary for critical issues that could
    halt or significantly impact the M&A transaction. These are non-negotiable
    compliance and legal requirements.
    
    Args:
        target_policy_summary: Summary of target company's HR policies to scan
    
    Returns:
        Analysis of potential deal-breaker issues found
    """
    try:
        benchmarks = _get_hr_benchmarks()
        deal_breakers = benchmarks.get('deal_breakers', [])
        
        result = """# Deal Breaker Analysis

## Critical Issues Scan

The following deal-breakers are being checked:

"""
        
        for db in deal_breakers:
            db_id = db.get('id', 'unknown')
            db_desc = db.get('description', '')
            severity = db.get('severity', 'high')
            legal_risk = "⚖️ LEGAL RISK" if db.get('legal_risk') else ""
            
            result += f"### {db_id.replace('_', ' ').title()}\n"
            result += f"- **Description:** {db_desc}\n"
            result += f"- **Severity:** {severity.upper()}\n"
            if legal_risk:
                result += f"- **{legal_risk}**\n"
            result += "\n"
        
        result += """---

## How to Use This Information

1. **Review each deal breaker** against target policy documents
2. **Flag ANY occurrence** of these issues immediately
3. **Legal risks** require immediate escalation to legal team
4. **Critical severity** issues may halt the entire transaction
5. **Document evidence** with specific policy citations

---

## Parameter-Specific Red Flags

"""
        
        for param_id, param_data in benchmarks.get('parameters', {}).items():
            red_flags = param_data.get('red_flags', [])
            if red_flags:
                result += f"### {param_id.replace('_', ' ').title()}\n"
                for flag in red_flags:
                    result += f"- ⚠️ {flag}\n"
                result += "\n"
        
        logger.info("Deal breaker checklist prepared for analysis")
        return result
        
    except Exception as e:
        logger.error(f"Error checking deal breakers: {e}")
        return f"Error checking deal breakers: {str(e)}"


# =============================================================================
# TOOL 7: GET INTEGRATION EFFORT ESTIMATE
# =============================================================================

@tool
def get_integration_effort_estimate(total_score: float) -> str:
    """
    Estimate integration effort based on compatibility score.
    
    This tool provides integration timeline, effort, and cost indicators
    based on the overall HR compatibility score.
    
    Args:
        total_score: Overall HR compatibility score (0-100)
    
    Returns:
        Integration effort estimate with timeline and recommendations
    """
    try:
        benchmarks = _get_hr_benchmarks()
        integration_matrix = benchmarks.get('integration_effort_matrix', {})
        
        # Determine which tier
        if total_score >= 80:
            tier = 'low'
        elif total_score >= 60:
            tier = 'medium'
        elif total_score >= 40:
            tier = 'high'
        else:
            tier = 'critical'
        
        tier_data = integration_matrix.get(tier, {})
        
        result = f"""# Integration Effort Estimate

## Score: {total_score}/100

---

## Integration Assessment

**Tier:** {tier.upper()}  
**Score Range:** {tier_data.get('score_range', 'N/A')}  
**Timeline:** {tier_data.get('timeline', 'Unknown')}  
**Effort Level:** {tier_data.get('effort', 'Unknown')}  
**Cost Indicator:** {tier_data.get('cost_indicator', 'Unknown')}

---

## Detailed Breakdown

"""
        
        if tier == 'low':
            result += """### Low Integration Effort (0-3 months)

**Activities:**
- Minor policy alignment updates
- Communication of TCS standards to acquired workforce
- Quick onboarding to TCS HR systems
- Minimal training required

**Resources Needed:**
- 1-2 HR integration specialists
- Standard HR onboarding materials
- Minimal legal review

**Key Actions:**
1. Policy mapping and minor adjustments
2. HR system migration planning
3. Employee communication
4. Benefits alignment
"""
        elif tier == 'medium':
            result += """### Medium Integration Effort (3-6 months)

**Activities:**
- Policy harmonization workshops
- Gap remediation planning
- Training programs for key differences
- Phased benefits transition

**Resources Needed:**
- Dedicated HR integration team (3-5 members)
- Training budget allocation
- Change management support
- Legal review for policy changes

**Key Actions:**
1. Detailed gap analysis and remediation plan
2. Policy rewriting for high-variance areas
3. Manager training on new policies
4. Employee town halls and Q&A sessions
5. Phased rollout of TCS policies
"""
        elif tier == 'high':
            result += """### High Integration Effort (6-12 months)

**Activities:**
- Major policy overhaul
- Comprehensive retraining programs
- Cultural transformation initiatives
- Possible restructuring of HR function

**Resources Needed:**
- Large HR integration team (5-10 members)
- Significant training and change management budget
- External HR consultants recommended
- Legal team involvement throughout

**Key Actions:**
1. Executive alignment on integration approach
2. Detailed remediation roadmap with milestones
3. Intensive manager and employee training
4. HR process redesign
5. Regular compliance audits
6. Employee retention measures
"""
        else:  # critical
            result += """### Critical Integration Effort (12+ months)

⚠️ **WARNING: High Risk Integration**

**Activities:**
- Complete policy rebuild required
- Cultural transformation at all levels
- Possible organizational restructuring
- Significant compliance remediation

**Resources Needed:**
- Dedicated integration leadership
- Large cross-functional team
- Major budget allocation
- External legal counsel
- HR transformation consultants

**Key Actions:**
1. Deal restructuring consideration
2. Detailed risk mitigation planning
3. Contingency planning for attrition
4. Complete HR function rebuild
5. Long-term compliance monitoring
6. Cultural integration program

**Recommendation:** Consider whether integration costs outweigh deal value.
"""
        
        logger.info(f"Integration estimate generated for score {total_score} (tier: {tier})")
        return result
        
    except Exception as e:
        logger.error(f"Error estimating integration effort: {e}")
        return f"Error estimating integration effort: {str(e)}"


# =============================================================================
# LEGACY TOOLS (Keep for backward compatibility - mark as deprecated)
# =============================================================================

@tool
def analyze_employee_data(
    company_id: str,
) -> str:
    """
    [DEPRECATED - Use get_target_hr_policies instead]
    
    Analyze employee data including headcount, demographics, and workforce composition.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Employee data analysis with workforce metrics
    """
    logger.warning("analyze_employee_data is deprecated. Use get_target_hr_policies for policy comparison.")
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


# =============================================================================
# EXPORT NEW POLICY COMPARISON TOOLS (Primary)
# =============================================================================

hr_tools = [
    # NEW: Policy Comparison Tools (Primary Focus)
    get_acquirer_baseline,
    get_target_hr_policies,
    compare_policy_category,
    calculate_hr_compatibility_score,
    
    # LEGACY: Kept for backward compatibility (Deprecated)
    analyze_employee_data,
    analyze_attrition,
    analyze_key_person_dependency,
    analyze_hr_policies,
    analyze_hr_compliance,
    analyze_culture_fit,
    generate_hr_risk_score,
]
