# HR Agent - M&A Policy Comparison

## Overview

The HR Agent has been redesigned to function as the **Acquirer's HR Policy Comparator** for M&A due diligence. It represents **TCS (the acquiring company)** and compares target company HR policies against TCS baseline standards.

## Key Changes from Original Design

### Before (General HR Due Diligence)
- Analyzed employee metrics, attrition, key person risk
- Generic HR analysis without baseline comparison
- No structured scoring framework

### After (Acquirer-Focused Policy Comparison)
- **Represents TCS as acquirer** with defined baseline standards
- **Compares target policies** against TCS policy manual
- **Scores 10 policy categories** (0-100 scale)
- **Identifies specific gaps** and integration requirements
- **Provides actionable recommendations** for policy alignment

---

## Architecture

### Files Created/Modified

```
src/hr_agent/
├── knowledge/                          # NEW FOLDER
│   ├── tcs_hr_policy.md               # TCS HR policy baseline (gold standard)
│   ├── hr_rubric_tcs_baseline.md      # Human-readable scoring rubric
│   ├── meta_prompt_tcs_baseline.md    # Meta-prompt for agent behavior
│   └── hr_benchmarks.json             # Machine-readable scoring parameters
├── state.py                            # UPDATED - Added policy comparison models
├── prompts.py                          # NEW - Chain of Thought prompts
├── tools.py                            # UPDATED - Policy comparison tools
└── graph.py                            # UPDATED - Uses new prompts and tools
```

---

## 10 Policy Parameters

Each parameter is weighted and scored 0-5:

| # | Parameter | Weight | Description |
|---|-----------|--------|-------------|
| 1 | Working Hours & Compensation | 10 | Hours, overtime, pay fairness |
| 2 | Leave & Time-Off | 15 | Annual, sick, maternity, paternity leave |
| 3 | Compensation Transparency | 12 | Pay structure, bonuses, benefits |
| 4 | Employment Terms | 10 | Contracts, probation, notice periods |
| 5 | Performance Management | 10 | Reviews, promotions, PIP process |
| 6 | Employee Relations & Culture | 12 | Grievance, DEI, engagement |
| 7 | Legal & Compliance | 13 | POSH, PF/ESI, labor laws |
| 8 | Exit & Separation | 8 | Severance, F&F, layoff terms |
| 9 | Data Privacy & Confidentiality | 5 | Data protection, monitoring |
| 10 | Training & Development | 5 | Learning programs, career paths |

**Total:** 100 points

---

## Scoring System

### Individual Parameter Scores (0-5)

| Score | Meaning | Description |
|-------|---------|-------------|
| **5** | Exceeds | Target exceeds or strongly matches TCS standard |
| **4** | Meets | Target fully meets TCS standard |
| **3** | Partial | Target partially meets (acceptable gaps) |
| **2** | Below | Target significantly below TCS standard |
| **1** | Minimal | Major gaps exist, unclear policy |
| **0** | Missing | No policy found or critical violation |

### Overall Risk Levels (0-100)

| Score Range | Risk Level | Description | Recommendation |
|-------------|------------|-------------|----------------|
| 80-100 | **Low** | Strong alignment with TCS | ✅ PROCEED |
| 60-79 | **Medium** | Moderate alignment | ⚠️ PROCEED WITH CAUTION |
| 40-59 | **High** | Significant gaps | ⚠️ CONDITIONAL |
| 0-39 | **Critical** | Major incompatibility | 🔴 REJECT OR RESTRUCTURE |

---

## TCS Baseline Highlights

Key standards from TCS HR Policy Manual (Version 2.0, December 2025):

### Leave Policy
- **Earned Leave:** 24 days
- **Casual Leave:** 8 days
- **Sick Leave:** 12 days
- **Maternity Leave:** 180 days (26 weeks)
- **Paternity Leave:** 14 days
- **Adoption Leave:** 180 days

### Working Hours
- **Weekly Hours:** 40
- **Standard Hours:** 9 AM - 6 PM
- **Core Hours:** 10 AM - 4 PM (mandatory)
- **Overtime Rate:** 2x base pay

### Compliance
- **POSH:** Zero tolerance, formal ICC
- **PF/ESI:** Fully compliant
- **Grievance:** 3-tier process with anti-retaliation

### Compensation
- **Structure:** Fixed + Variable
- **Review:** Annual
- **Benefits:** Medical, PF, Gratuity, ESOP

---

## New Tools

### 1. `get_acquirer_baseline()`
Loads TCS HR policy baseline with all parameters and standards.

**Returns:** Complete TCS policy manual + scoring parameters

### 2. `get_target_hr_policies(company_id)`
Retrieves ALL HR policy documents for target company from RAG.

**Args:** 
- `company_id`: Target company (BBD, XYZ, SUPERNOVA, etc.)

**Returns:** All HR handbooks, policies, and related documents

### 3. `compare_policy_category(category_name, acquirer_data, target_data)`
Compares one of 10 policy categories between TCS and target.

**Args:**
- `category_name`: e.g., "leave_time_off", "legal_compliance"
- `acquirer_data`: TCS baseline (from tool #1)
- `target_data`: Target policies (from tool #2)

**Returns:** Comparison framework with scoring guidance

### 4. `calculate_hr_compatibility_score(comparison_data)`
Calculates overall score from individual parameter scores.

**Args:**
- `comparison_data`: JSON with all parameter scores

**Returns:** Overall score, risk level, recommendation

---

## Usage Example

### Query Format

```
Analyze the HR policies of XYZ Ltd for acquisition by TCS.

Compare XYZ's HR policies against TCS baseline standards and provide:
1. Overall compatibility score (0-100)
2. Category breakdown for all 10 parameters
3. Policy gaps and red flags
4. Integration recommendations
```

### Agent Workflow

1. **Load TCS Baseline** → `get_acquirer_baseline()`
2. **Retrieve Target Policies** → `get_target_hr_policies("XYZ")`
3. **Compare Each Category** → `compare_policy_category()` x10
4. **Calculate Score** → `calculate_hr_compatibility_score()`
5. **Generate Report** → Structured output with recommendations

### Expected Output

```markdown
# HR Policy Compatibility Report

## Executive Summary
**Target Company:** XYZ Ltd
**Acquirer:** TCS
**Overall Score:** 72/100
**Risk Level:** Medium
**Recommendation:** PROCEED WITH CAUTION

## Category Breakdown
| Category | Score | Status |
|----------|-------|--------|
| Working Hours & Compensation | 8/10 | 🟢 |
| Leave & Time-Off | 10/15 | 🟡 |
| Compensation Transparency | 10/12 | 🟢 |
| ... | ... | ... |

## Policy Gaps Identified
1. ⚠️ **Maternity Leave:** XYZ offers 12 weeks vs TCS 26 weeks
2. ⚠️ **POSH Committee:** No ICC documented
3. ℹ️ **Remote Work:** No formal policy found

## Integration Recommendations
1. Adopt TCS leave structure during Day 1 integration
2. Establish ICC for POSH compliance before close
3. Create remote work policy (3-month timeline)
```

---

## Testing

### Run Test Script

```bash
cd backend
python scripts/test_hr_agent.py
```

This will:
1. Test individual tools
2. Run full policy comparison for BBD company
3. Display results and compatibility score

### Manual Testing with LangGraph Studio

```bash
langgraph dev
```

Then query:
```
Analyze BBD's HR policies for acquisition by TCS
```

---

## Deal Breakers

These issues automatically trigger REJECT recommendation:

- ❌ POSH compliance missing (no ICC)
- ❌ Labor law violations documented
- ❌ Below minimum wage payments
- ❌ Toxic culture evidence (multiple harassment claims)
- ❌ No grievance mechanism
- ❌ Discriminatory practices
- ❌ Child labor or forced labor
- ❌ Systematic overtime violations

---

## Integration Recommendations

HR Agent provides specific, actionable recommendations:

### Immediate (Pre-Close)
- Resolve deal breakers
- Get POSH compliance certification
- Document critical policy gaps

### Short-Term (0-3 months)
- Harmonize leave policies
- Implement TCS grievance process
- Establish data privacy framework

### Medium-Term (3-12 months)
- Full policy alignment
- Cultural integration programs
- Performance management adoption

---

## Configuration

### Customize TCS Baseline

Edit: `src/hr_agent/knowledge/tcs_hr_policy.md`

### Adjust Weights

Edit: `src/hr_agent/knowledge/hr_benchmarks.json`

```json
{
  "parameters": {
    "leave_time_off": {
      "weight": 15,  // Change this
      ...
    }
  }
}
```

### Modify Scoring Thresholds

```json
{
  "risk_thresholds": {
    "low": {
      "min_score": 80,  // Adjust threshold
      "max_score": 100
    }
  }
}
```

---

## Legacy Tools

Previous HR Agent tools are still available but marked as **DEPRECATED**:
- `analyze_employee_data()`
- `analyze_attrition()`
- `analyze_key_person_dependency()`
- `analyze_hr_compliance()`
- `analyze_culture_fit()`
- `generate_hr_risk_score()`

These will log warnings and should not be used for new policy comparison workflows.

---

## Next Steps

1. **Test with Real Data:** Run against all available companies (BBD, XYZ, Supernova)
2. **Refine Prompts:** Adjust CoT prompts based on LLM performance
3. **Add More Parameters:** Extend beyond 10 if needed
4. **Integration with Supervisor:** Ensure supervisor agent routes correctly
5. **Create Visualizations:** Build UI for score visualization

---

## API Integration

The HR Agent can be invoked via:

1. **LangGraph Studio:** Visual interface
2. **LangGraph Server:** REST API
3. **Python SDK:** Direct invocation
4. **Supervisor Agent:** Orchestrated workflow

Example Python invocation:

```python
from langchain_core.messages import HumanMessage
from src.hr_agent.graph import hr_agent

result = hr_agent.invoke({
    "messages": [HumanMessage(content="Analyze BBD HR policies for TCS acquisition")],
    "target_company": {"company_name": "BBD", "company_id": "BBD"}
})

print(result["messages"][-1].content)
```

---

## Support

For questions or issues:
1. Check test script: `scripts/test_hr_agent.py`
2. Review prompts: `src/hr_agent/prompts.py`
3. Inspect tools: `src/hr_agent/tools.py`
4. Examine state models: `src/hr_agent/state.py`

---

**Last Updated:** December 6, 2025  
**Version:** 2.0 (Policy Comparison Focus)
