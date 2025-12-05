#!/usr/bin/env python3
"""
Setup script for initializing the M&A Due Diligence Orchestrator data.

This script:
1. Creates necessary directories
2. Generates synthetic company data
3. Loads documents into ChromaDB vector stores
4. Validates the setup

Usage:
    python scripts/setup_data.py
    python scripts/setup_data.py --force  # Regenerate all data
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

# Company configurations
COMPANIES = [
    {
        "id": "TECHCORP",
        "name": "TechCorp Solutions Inc.",
        "industry": "Technology",
        "founded": 2015,
        "employees": 500,
        "revenue": 75000000
    },
    {
        "id": "FINSERV",
        "name": "FinServ Holdings LLC",
        "industry": "Financial Services",
        "founded": 2008,
        "employees": 1200,
        "revenue": 150000000
    },
    {
        "id": "HEALTHTECH",
        "name": "HealthTech Innovations",
        "industry": "Healthcare Technology",
        "founded": 2018,
        "employees": 250,
        "revenue": 35000000
    },
    {
        "id": "RETAILMAX",
        "name": "RetailMax Corporation",
        "industry": "Retail",
        "founded": 2005,
        "employees": 3000,
        "revenue": 500000000
    },
    {
        "id": "GREENERGY",
        "name": "GreenErgy Power Systems",
        "industry": "Clean Energy",
        "founded": 2012,
        "employees": 400,
        "revenue": 95000000
    }
]


def create_directories():
    """Create necessary directories for data storage."""
    print("📁 Creating directories...")
    
    base_path = Path(__file__).parent.parent
    
    directories = [
        base_path / "data" / "raw",
        base_path / "data" / "processed",
        base_path / "data" / "chroma_db",
        base_path / "data" / "companies",
        base_path / "logs",
        base_path / "output"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory.relative_to(base_path)}")
    
    return True


def generate_synthetic_data(force: bool = False):
    """Generate synthetic company data."""
    print("\n📊 Generating synthetic company data...")
    
    try:
        from synthetic_data_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        
        for company in COMPANIES:
            print(f"  Generating data for {company['name']}...")
            
            # Generate company data
            data = generator.generate_company_data(
                company_id=company["id"],
                company_name=company["name"],
                industry=company["industry"]
            )
            
            print(f"    ✅ Generated {len(data.get('financial', []))} financial records")
            print(f"    ✅ Generated {len(data.get('legal', []))} legal documents")
            print(f"    ✅ Generated {len(data.get('hr', []))} HR records")
        
        print("\n✅ Synthetic data generation complete!")
        return True
        
    except ImportError as e:
        print(f"  ⚠️ Import error: {e}")
        print("  Using fallback data generation...")
        return generate_fallback_data()
    except Exception as e:
        print(f"  ❌ Error generating data: {e}")
        return False


def generate_fallback_data():
    """Generate minimal fallback data if main generator fails."""
    print("\n📊 Generating fallback synthetic data...")
    
    import json
    base_path = Path(__file__).parent.parent / "data" / "companies"
    
    for company in COMPANIES:
        company_path = base_path / company["id"]
        company_path.mkdir(parents=True, exist_ok=True)
        
        # Generate minimal financial data
        financial_data = {
            "company_id": company["id"],
            "company_name": company["name"],
            "industry": company["industry"],
            "financial_statements": [
                {
                    "year": 2024,
                    "revenue": company["revenue"],
                    "gross_profit": company["revenue"] * 0.4,
                    "operating_income": company["revenue"] * 0.15,
                    "net_income": company["revenue"] * 0.1,
                    "total_assets": company["revenue"] * 1.5,
                    "total_liabilities": company["revenue"] * 0.6,
                    "total_equity": company["revenue"] * 0.9,
                    "current_assets": company["revenue"] * 0.5,
                    "current_liabilities": company["revenue"] * 0.25
                }
            ],
            "metrics": {
                "gross_margin": 0.4,
                "operating_margin": 0.15,
                "net_margin": 0.1,
                "current_ratio": 2.0,
                "debt_to_equity": 0.67,
                "roe": 0.11
            }
        }
        
        with open(company_path / "financial_data.json", "w") as f:
            json.dump(financial_data, f, indent=2)
        
        # Generate minimal legal data
        legal_data = {
            "company_id": company["id"],
            "litigation": [
                {
                    "case_id": f"{company['id']}-LIT-001",
                    "type": "Contract Dispute",
                    "status": "Pending",
                    "exposure": 500000,
                    "probability": 0.3
                }
            ],
            "contracts": [
                {
                    "contract_id": f"{company['id']}-CON-001",
                    "type": "Customer Agreement",
                    "value": company["revenue"] * 0.1,
                    "term_years": 3,
                    "change_of_control": True
                }
            ],
            "ip_portfolio": [
                {
                    "ip_id": f"{company['id']}-IP-001",
                    "type": "Patent",
                    "status": "Active",
                    "expiry_year": 2030
                }
            ],
            "compliance": {
                "status": "Compliant",
                "last_audit": "2024-01-15",
                "findings": []
            }
        }
        
        with open(company_path / "legal_data.json", "w") as f:
            json.dump(legal_data, f, indent=2)
        
        # Generate minimal HR data
        hr_data = {
            "company_id": company["id"],
            "headcount": company["employees"],
            "attrition_rate": 0.12,
            "departments": [
                {"name": "Engineering", "headcount": int(company["employees"] * 0.4)},
                {"name": "Sales", "headcount": int(company["employees"] * 0.25)},
                {"name": "Operations", "headcount": int(company["employees"] * 0.2)},
                {"name": "Finance", "headcount": int(company["employees"] * 0.1)},
                {"name": "HR", "headcount": int(company["employees"] * 0.05)}
            ],
            "key_persons": [
                {"name": "John Smith", "role": "CEO", "tenure_years": 5, "risk_score": 0.3},
                {"name": "Jane Doe", "role": "CTO", "tenure_years": 4, "risk_score": 0.4}
            ],
            "culture_metrics": {
                "satisfaction_score": 3.8,
                "engagement_score": 4.0,
                "nps": 45
            }
        }
        
        with open(company_path / "hr_data.json", "w") as f:
            json.dump(hr_data, f, indent=2)
        
        print(f"  ✅ Generated fallback data for {company['name']}")
    
    return True


def load_documents_to_vectorstore():
    """Load documents into ChromaDB."""
    print("\n📚 Loading documents to ChromaDB...")
    
    try:
        from document_loader import DocumentLoader
        
        loader = DocumentLoader()
        
        print("  Loading financial documents...")
        loader.load_financial_documents()
        print("  ✅ Financial documents loaded")
        
        print("  Loading legal documents...")
        loader.load_legal_documents()
        print("  ✅ Legal documents loaded")
        
        print("  Loading HR documents...")
        loader.load_hr_documents()
        print("  ✅ HR documents loaded")
        
        print("\n✅ All documents loaded to ChromaDB!")
        return True
        
    except ImportError as e:
        print(f"  ⚠️ Import error: {e}")
        print("  Skipping vector store loading...")
        return True  # Not critical for basic demo
    except Exception as e:
        print(f"  ❌ Error loading documents: {e}")
        return False


def validate_setup():
    """Validate that the setup is complete."""
    print("\n🔍 Validating setup...")
    
    base_path = Path(__file__).parent.parent
    all_valid = True
    
    # Check directories
    required_dirs = [
        "data/companies",
        "data/chroma_db"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"  ✅ Directory exists: {dir_path}")
        else:
            print(f"  ❌ Missing directory: {dir_path}")
            all_valid = False
    
    # Check company data files
    for company in COMPANIES:
        company_path = base_path / "data" / "companies" / company["id"]
        if company_path.exists():
            print(f"  ✅ Company data exists: {company['id']}")
        else:
            print(f"  ⚠️ Missing company data: {company['id']}")
    
    # Check environment
    env_file = base_path / ".env"
    if env_file.exists():
        print("  ✅ Environment file (.env) exists")
    else:
        print("  ⚠️ Missing .env file - copy from .env.example")
    
    return all_valid


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup M&A Due Diligence Orchestrator data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all data"
    )
    parser.add_argument(
        "--skip-vectorstore",
        action="store_true",
        help="Skip loading to vector store"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  M&A Due Diligence Orchestrator - Data Setup")
    print("=" * 60)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Create directories
    if not create_directories():
        print("\n❌ Failed to create directories")
        sys.exit(1)
    
    # Step 2: Generate synthetic data
    if not generate_synthetic_data(force=args.force):
        print("\n❌ Failed to generate synthetic data")
        sys.exit(1)
    
    # Step 3: Load to vector store
    if not args.skip_vectorstore:
        if not load_documents_to_vectorstore():
            print("\n⚠️ Vector store loading failed, but continuing...")
    
    # Step 4: Validate
    if validate_setup():
        print("\n" + "=" * 60)
        print("  ✅ Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Ensure .env file has your API keys configured")
        print("  2. Run: langgraph dev --allow-blocking")
        print("  3. Open the LangChain Chat UI in your browser")
        print("  4. Try: 'Analyze TECHCORP for potential acquisition'")
    else:
        print("\n⚠️ Setup completed with warnings")
        sys.exit(0)


if __name__ == "__main__":
    main()
