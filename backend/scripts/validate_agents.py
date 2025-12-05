#!/usr/bin/env python3
"""
Validation script to verify all agent modules are correctly configured.

Usage:
    python scripts/validate_agents.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

def validate_imports():
    """Validate all module imports work correctly."""
    print("=" * 60)
    print("M&A Due Diligence Orchestrator - Agent Validation")
    print("=" * 60)
    
    errors = []
    
    # Test config imports
    print("\n📋 Testing Configuration...")
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        print(f"  ✅ Settings loaded (ChromaDB: {settings.chroma_persist_directory})")
    except Exception as e:
        errors.append(f"Config settings: {e}")
        print(f"  ❌ Settings failed: {e}")
    
    try:
        from src.config.llm_config import get_llm, get_embedding_model
        print("  ✅ LLM config module loaded")
    except Exception as e:
        errors.append(f"LLM config: {e}")
        print(f"  ❌ LLM config failed: {e}")
    
    # Test common imports
    print("\n📋 Testing Common Utilities...")
    try:
        from src.common.state import BaseAgentState, CompanyInfo, RiskScore
        print("  ✅ Common state loaded")
    except Exception as e:
        errors.append(f"Common state: {e}")
        print(f"  ❌ Common state failed: {e}")
    
    try:
        from src.common.utils import format_currency, calculate_risk_score
        print("  ✅ Common utils loaded")
    except Exception as e:
        errors.append(f"Common utils: {e}")
        print(f"  ❌ Common utils failed: {e}")
    
    try:
        from src.common.logging_config import get_logger
        print("  ✅ Logging config loaded")
    except Exception as e:
        errors.append(f"Logging: {e}")
        print(f"  ❌ Logging failed: {e}")
    
    # Test RAG Agent
    print("\n📋 Testing RAG Agent...")
    try:
        from src.rag_agent.tools import (
            retrieve_financial_documents,
            retrieve_legal_documents,
            retrieve_hr_documents,
            retrieve_employee_records,
            retrieve_contracts,
            retrieve_litigation_records,
            search_all_documents,
            get_company_overview,
            COLLECTIONS,
            COMPANIES,
        )
        print(f"  ✅ RAG tools loaded ({len(COLLECTIONS)} collections, {len(COMPANIES)} companies)")
    except Exception as e:
        errors.append(f"RAG tools: {e}")
        print(f"  ❌ RAG tools failed: {e}")
    
    try:
        from src.rag_agent.state import RAGAgentState
        print("  ✅ RAG state loaded")
    except Exception as e:
        errors.append(f"RAG state: {e}")
        print(f"  ❌ RAG state failed: {e}")
    
    # Test Finance Agent
    print("\n📋 Testing Finance Agent...")
    try:
        from src.finance_agent.tools import (
            analyze_balance_sheet,
            analyze_income_statement,
            analyze_cash_flow,
            analyze_financial_ratios,
            assess_financial_risk,
            compare_financial_performance,
            finance_tools,
        )
        print(f"  ✅ Finance tools loaded ({len(finance_tools)} tools)")
    except Exception as e:
        errors.append(f"Finance tools: {e}")
        print(f"  ❌ Finance tools failed: {e}")
    
    try:
        from src.finance_agent.state import FinanceAgentState
        print("  ✅ Finance state loaded")
    except Exception as e:
        errors.append(f"Finance state: {e}")
        print(f"  ❌ Finance state failed: {e}")
    
    # Test Legal Agent
    print("\n📋 Testing Legal Agent...")
    try:
        from src.legal_agent.tools import (
            analyze_contracts,
            analyze_litigation_exposure,
            analyze_ip_portfolio,
            analyze_regulatory_compliance,
            analyze_corporate_governance,
            generate_legal_risk_score,
            legal_tools,
        )
        print(f"  ✅ Legal tools loaded ({len(legal_tools)} tools)")
    except Exception as e:
        errors.append(f"Legal tools: {e}")
        print(f"  ❌ Legal tools failed: {e}")
    
    try:
        from src.legal_agent.state import LegalAgentState
        print("  ✅ Legal state loaded")
    except Exception as e:
        errors.append(f"Legal state: {e}")
        print(f"  ❌ Legal state failed: {e}")
    
    # Test HR Agent
    print("\n📋 Testing HR Agent...")
    try:
        from src.hr_agent.tools import (
            analyze_employee_data,
            analyze_attrition,
            analyze_key_person_dependency,
            analyze_hr_policies,
            analyze_hr_compliance,
            analyze_culture_fit,
            generate_hr_risk_score,
            hr_tools,
        )
        print(f"  ✅ HR tools loaded ({len(hr_tools)} tools)")
    except Exception as e:
        errors.append(f"HR tools: {e}")
        print(f"  ❌ HR tools failed: {e}")
    
    try:
        from src.hr_agent.state import HRAgentState
        print("  ✅ HR state loaded")
    except Exception as e:
        errors.append(f"HR state: {e}")
        print(f"  ❌ HR state failed: {e}")
    
    # Test Analyst Agent
    print("\n📋 Testing Analyst Agent...")
    try:
        from src.analyst_agent.tools import (
            analyze_target_company,
            estimate_synergies,
            consolidate_due_diligence,
            generate_deal_recommendation,
            compare_acquisition_targets,
            analyst_tools,
        )
        print(f"  ✅ Analyst tools loaded ({len(analyst_tools)} tools)")
    except Exception as e:
        errors.append(f"Analyst tools: {e}")
        print(f"  ❌ Analyst tools failed: {e}")
    
    try:
        from src.analyst_agent.state import AnalystAgentState
        print("  ✅ Analyst state loaded")
    except Exception as e:
        errors.append(f"Analyst state: {e}")
        print(f"  ❌ Analyst state failed: {e}")
    
    # Test Supervisor
    print("\n📋 Testing Supervisor...")
    try:
        from src.supervisor.state import SupervisorState
        print("  ✅ Supervisor state loaded")
    except Exception as e:
        errors.append(f"Supervisor state: {e}")
        print(f"  ❌ Supervisor state failed: {e}")
    
    try:
        from src.supervisor.prompts import SUPERVISOR_SYSTEM_PROMPT
        print("  ✅ Supervisor prompts loaded")
    except Exception as e:
        errors.append(f"Supervisor prompts: {e}")
        print(f"  ❌ Supervisor prompts failed: {e}")
    
    # Test Document Loader
    print("\n📋 Testing Document Loader...")
    try:
        from data.document_loader import (
            DocumentLoader,
            load_all_documents,
            COLLECTIONS,
            COMPANIES,
            RAW_DATA_PATH,
        )
        print(f"  ✅ Document loader loaded")
        print(f"     Raw data path: {RAW_DATA_PATH}")
        print(f"     Path exists: {RAW_DATA_PATH.exists()}")
    except Exception as e:
        errors.append(f"Document loader: {e}")
        print(f"  ❌ Document loader failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"   - {err}")
        return False
    else:
        print("✅ All validations PASSED!")
        print("\nNext steps:")
        print("  1. Set up environment variables in .env")
        print("  2. Run: python data/document_loader.py")
        print("  3. Run: langgraph dev")
        return True


def check_data_files():
    """Check if data files exist."""
    print("\n📋 Checking Data Files...")
    
    data_path = Path(__file__).parent.parent / "data" / "row_data"
    
    if not data_path.exists():
        print(f"  ⚠️ Data directory not found: {data_path}")
        return
    
    categories = ["Finance", "HR Data", "legal"]
    for cat in categories:
        cat_path = data_path / cat
        if cat_path.exists():
            files = list(cat_path.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            print(f"  ✅ {cat}: {file_count} files found")
        else:
            print(f"  ⚠️ {cat}: directory not found")


if __name__ == "__main__":
    success = validate_imports()
    check_data_files()
    
    print("\n" + "=" * 60)
    sys.exit(0 if success else 1)
