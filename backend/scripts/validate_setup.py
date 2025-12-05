#!/usr/bin/env python3
"""
Quick validation test for the M&A Due Diligence Orchestrator.

This script tests basic imports and component initialization without
requiring API keys or external services.

Usage:
    python scripts/validate_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

def test_import(module_name: str, component: str = None) -> bool:
    """Test importing a module."""
    try:
        if component:
            exec(f"from {module_name} import {component}")
        else:
            exec(f"import {module_name}")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️ Import warning for {module_name}: {e}")
        return True  # Non-import errors are okay


def main():
    """Run validation tests."""
    print("=" * 60)
    print("  M&A Due Diligence Orchestrator - Setup Validation")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test 1: Core dependencies
    print("📦 Testing core dependencies...")
    core_deps = [
        ("pydantic", None),
        ("structlog", None),
    ]
    
    for module, component in core_deps:
        if test_import(module, component):
            print(f"  ✅ {module}")
        else:
            all_passed = False
    
    # Test 2: LangChain/LangGraph (optional - may not be installed)
    print("\n📦 Testing LangChain/LangGraph dependencies...")
    lang_deps = [
        ("langchain_core", None),
        ("langgraph", None),
    ]
    
    for module, component in lang_deps:
        if test_import(module, component):
            print(f"  ✅ {module}")
        else:
            print(f"  ⚠️ {module} not installed (install with: pip install -e .)")
    
    # Test 3: Project modules (config)
    print("\n📦 Testing configuration modules...")
    config_modules = [
        ("src.config.settings", "Settings"),
        ("src.config.llm_config", "get_llm"),
    ]
    
    for module, component in config_modules:
        try:
            exec(f"from {module} import {component}")
            print(f"  ✅ {module}.{component}")
        except Exception as e:
            print(f"  ⚠️ {module}.{component}: {e}")
    
    # Test 4: Common modules
    print("\n📦 Testing common modules...")
    common_modules = [
        ("src.common.state", "CompanyInfo"),
        ("src.common.errors", "DueDiligenceError"),
        ("src.common.utils", "format_currency"),
        ("src.common.guardrails", "PIIFilter"),
    ]
    
    for module, component in common_modules:
        try:
            exec(f"from {module} import {component}")
            print(f"  ✅ {module}.{component}")
        except Exception as e:
            print(f"  ⚠️ {module}.{component}: {e}")
    
    # Test 5: Agent modules
    print("\n📦 Testing agent modules...")
    agent_modules = [
        "src.finance_agent.state",
        "src.legal_agent.state",
        "src.hr_agent.state",
        "src.analyst_agent.state",
        "src.rag_agent.state",
        "src.supervisor.state",
    ]
    
    for module in agent_modules:
        try:
            exec(f"import {module}")
            agent_name = module.split(".")[1]
            print(f"  ✅ {agent_name}")
        except Exception as e:
            print(f"  ⚠️ {module}: {e}")
    
    # Test 6: Data modules
    print("\n📦 Testing data modules...")
    data_modules = [
        ("data.synthetic_data_generator", "SyntheticDataGenerator"),
        ("data.document_loader", "DocumentLoader"),
    ]
    
    for module, component in data_modules:
        try:
            exec(f"from {module} import {component}")
            print(f"  ✅ {module}.{component}")
        except Exception as e:
            print(f"  ⚠️ {module}.{component}: {e}")
    
    # Test 7: Directory structure
    print("\n📁 Checking directory structure...")
    base_path = Path(__file__).parent.parent
    
    required_dirs = [
        "src/config",
        "src/common",
        "src/supervisor",
        "src/finance_agent",
        "src/legal_agent",
        "src/hr_agent",
        "src/analyst_agent",
        "src/rag_agent",
        "data",
        "scripts",
        "tests",
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ Missing: {dir_path}/")
            all_passed = False
    
    # Test 8: Required files
    print("\n📄 Checking required files...")
    required_files = [
        "langgraph.json",
        "pyproject.toml",
        "README.md",
        ".env.example",
        "src/__init__.py",
        "src/supervisor/graph.py",
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ Missing: {file_path}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ All validation checks passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and add your API keys")
        print("  2. Install dependencies: pip install -e .")
        print("  3. Initialize data: python scripts/setup_data.py")
        print("  4. Start server: langgraph dev --allow-blocking")
    else:
        print("  ⚠️ Some validation checks failed")
        print("=" * 60)
        print("\nPlease resolve the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
