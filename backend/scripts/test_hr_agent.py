"""
Test Script for HR Agent - M&A Policy Comparison

This script demonstrates the new HR Agent functionality that compares
target company HR policies against TCS (acquirer) baseline standards.

Usage:
    python scripts/test_hr_agent.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from langchain_core.messages import HumanMessage
from src.hr_agent.graph import hr_agent
from src.config.llm_config import get_llm


def test_hr_policy_comparison(target_company: str = "BBD"):
    """
    Test HR Agent's policy comparison functionality.
    
    Args:
        target_company: Target company to analyze (BBD, XYZ, SUPERNOVA, etc.)
    """
    print("=" * 80)
    print(f"HR AGENT TEST: Policy Comparison for {target_company}")
    print("=" * 80)
    print()
    
    # Create query for HR Agent
    query = f"""Analyze the HR policies of {target_company} for acquisition by TCS.

Compare {target_company}'s HR policies against TCS baseline standards and provide:
1. Overall compatibility score (0-100)
2. Category breakdown for all 10 parameters
3. Policy gaps and red flags
4. Integration recommendations

Focus on policy comparison only."""
    
    print(f"📝 Query: {query}")
    print()
    print("-" * 80)
    print()
    
    # Invoke HR Agent
    print("🤖 Invoking HR Agent...")
    print()
    
    try:
        result = hr_agent.invoke({
            "messages": [HumanMessage(content=query)],
            "target_company": {"company_name": target_company, "company_id": target_company}
        })
        
        # Extract final response
        if result.get("messages"):
            final_message = result["messages"][-1]
            print("=" * 80)
            print("HR AGENT RESPONSE:")
            print("=" * 80)
            print()
            print(final_message.content)
            print()
        else:
            print("❌ No response from HR Agent")
            
    except Exception as e:
        print(f"❌ Error running HR Agent: {e}")
        import traceback
        traceback.print_exc()


def test_individual_tools():
    """Test individual HR Agent tools."""
    print("=" * 80)
    print("TESTING INDIVIDUAL HR AGENT TOOLS")
    print("=" * 80)
    print()
    
    from src.hr_agent.tools import (
        get_acquirer_baseline,
        get_target_hr_policies,
        compare_policy_category,
        calculate_hr_compatibility_score
    )
    
    # Test 1: Get Acquirer Baseline
    print("📋 Test 1: Get Acquirer (TCS) Baseline")
    print("-" * 80)
    try:
        baseline = get_acquirer_baseline.invoke({})
        print(baseline[:1000] + "...\n")  # Print first 1000 chars
        print("✅ Baseline loaded successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()
    
    # Test 2: Get Target HR Policies
    print("📋 Test 2: Get Target HR Policies (BBD)")
    print("-" * 80)
    try:
        target_policies = get_target_hr_policies.invoke({"company_id": "BBD"})
        print(target_policies[:800] + "...\n")
        print("✅ Target policies retrieved successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()
    
    # Test 3: Compare Policy Category
    print("📋 Test 3: Compare Policy Category (Leave & Time-Off)")
    print("-" * 80)
    try:
        comparison = compare_policy_category.invoke({
            "category_name": "leave_time_off",
            "acquirer_data": "TCS baseline data",
            "target_data": "BBD target data"
        })
        print(comparison[:800] + "...\n")
        print("✅ Category comparison framework generated")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()
    
    # Test 4: Calculate Compatibility Score
    print("📋 Test 4: Calculate HR Compatibility Score")
    print("-" * 80)
    try:
        sample_data = {
            "parameters": [
                {"name": "working_hours_compensation", "score": 4, "weight": 10},
                {"name": "leave_time_off", "score": 3, "weight": 15},
                {"name": "compensation_transparency", "score": 4, "weight": 12},
                {"name": "employment_terms", "score": 3, "weight": 10},
                {"name": "performance_management", "score": 3, "weight": 10},
                {"name": "employee_relations_culture", "score": 4, "weight": 12},
                {"name": "legal_compliance", "score": 3, "weight": 13},
                {"name": "exit_separation", "score": 3, "weight": 8},
                {"name": "data_privacy_confidentiality", "score": 2, "weight": 5},
                {"name": "training_development", "score": 2, "weight": 5}
            ]
        }
        
        import json
        score_result = calculate_hr_compatibility_score.invoke({
            "comparison_data": json.dumps(sample_data)
        })
        print(score_result)
        print("✅ Compatibility score calculated successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()


def main():
    """Main test function."""
    print()
    print("🚀 HR AGENT - M&A POLICY COMPARISON TEST SUITE")
    print()
    
    # Test individual tools first
    test_individual_tools()
    
    print()
    print("=" * 80)
    print()
    
    # Test full agent workflow
    test_hr_policy_comparison(target_company="BBD")
    
    print()
    print("=" * 80)
    print("✅ HR AGENT TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
