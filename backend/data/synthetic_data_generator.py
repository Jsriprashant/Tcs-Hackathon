"""
Synthetic Data Generator for M&A Due Diligence Demo.

Generates realistic synthetic data for:
- Financial statements (5 companies, 5 years)
- Legal documents (contracts, litigations, IP)
- HR data (policies, employee records)
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Company profiles for synthetic data
COMPANIES = {
    "TECHCORP": {
        "name": "TechCorp Solutions Ltd",
        "industry": "Technology",
        "founded": 2010,
        "headquarters": "Bangalore, India",
        "employees": 2500,
        "base_revenue": 150_000_000,
        "growth_rate": 0.15,
        "risk_profile": "low",
    },
    "FINSERV": {
        "name": "FinServ Global Inc",
        "industry": "Financial Services",
        "founded": 2005,
        "headquarters": "Mumbai, India",
        "employees": 5000,
        "base_revenue": 300_000_000,
        "growth_rate": 0.08,
        "risk_profile": "medium",
    },
    "HEALTHTECH": {
        "name": "HealthTech Innovations",
        "industry": "Healthcare Technology",
        "founded": 2015,
        "headquarters": "Hyderabad, India",
        "employees": 800,
        "base_revenue": 50_000_000,
        "growth_rate": 0.25,
        "risk_profile": "medium",
    },
    "RETAILMAX": {
        "name": "RetailMax Enterprises",
        "industry": "Retail",
        "founded": 2000,
        "headquarters": "Delhi, India",
        "employees": 15000,
        "base_revenue": 500_000_000,
        "growth_rate": 0.03,
        "risk_profile": "high",
    },
    "GREENERGY": {
        "name": "GreenErgy Power Solutions",
        "industry": "Renewable Energy",
        "founded": 2012,
        "headquarters": "Chennai, India",
        "employees": 1200,
        "base_revenue": 80_000_000,
        "growth_rate": 0.20,
        "risk_profile": "low",
    },
}


def generate_financial_statements(
    company_id: str,
    years: list[int] = None,
) -> dict[str, Any]:
    """Generate synthetic financial statements for a company."""
    
    if years is None:
        years = list(range(2020, 2025))
    
    company = COMPANIES[company_id]
    base_revenue = company["base_revenue"]
    growth_rate = company["growth_rate"]
    risk_profile = company["risk_profile"]
    
    financials = {
        "company_id": company_id,
        "company_name": company["name"],
        "industry": company["industry"],
        "statements": []
    }
    
    for i, year in enumerate(years):
        # Calculate revenue with growth and some randomness
        revenue = base_revenue * ((1 + growth_rate) ** i)
        revenue *= random.uniform(0.95, 1.05)  # Add some variance
        
        # Generate income statement
        cogs = revenue * random.uniform(0.55, 0.65)
        gross_profit = revenue - cogs
        operating_expenses = revenue * random.uniform(0.20, 0.30)
        operating_income = gross_profit - operating_expenses
        interest_expense = revenue * random.uniform(0.01, 0.03)
        ebt = operating_income - interest_expense
        tax_rate = 0.25
        net_income = ebt * (1 - tax_rate) if ebt > 0 else ebt
        
        # Generate balance sheet
        total_assets = revenue * random.uniform(1.5, 2.0)
        current_assets = total_assets * random.uniform(0.35, 0.45)
        cash = current_assets * random.uniform(0.20, 0.40)
        accounts_receivable = current_assets * random.uniform(0.30, 0.40)
        inventory = current_assets - cash - accounts_receivable
        
        total_liabilities = total_assets * random.uniform(0.40, 0.60)
        current_liabilities = total_liabilities * random.uniform(0.30, 0.50)
        long_term_debt = total_liabilities - current_liabilities
        
        total_equity = total_assets - total_liabilities
        
        # Generate cash flow
        depreciation = total_assets * 0.05
        operating_cash_flow = net_income + depreciation
        if risk_profile == "high":
            operating_cash_flow *= random.uniform(0.6, 0.9)  # Lower for risky companies
        
        capex = -revenue * random.uniform(0.05, 0.10)
        investing_cash_flow = capex
        financing_cash_flow = -net_income * random.uniform(0.2, 0.4)  # Dividends, debt repayment
        
        statement = {
            "fiscal_year": year,
            "income_statement": {
                "revenue": round(revenue, 2),
                "cost_of_goods_sold": round(cogs, 2),
                "gross_profit": round(gross_profit, 2),
                "operating_expenses": round(operating_expenses, 2),
                "operating_income": round(operating_income, 2),
                "interest_expense": round(interest_expense, 2),
                "earnings_before_tax": round(ebt, 2),
                "income_tax": round(ebt * tax_rate if ebt > 0 else 0, 2),
                "net_income": round(net_income, 2),
                "ebitda": round(operating_income + depreciation, 2),
            },
            "balance_sheet": {
                "total_assets": round(total_assets, 2),
                "current_assets": round(current_assets, 2),
                "cash_and_equivalents": round(cash, 2),
                "accounts_receivable": round(accounts_receivable, 2),
                "inventory": round(inventory, 2),
                "fixed_assets": round(total_assets - current_assets, 2),
                "total_liabilities": round(total_liabilities, 2),
                "current_liabilities": round(current_liabilities, 2),
                "long_term_debt": round(long_term_debt, 2),
                "total_equity": round(total_equity, 2),
            },
            "cash_flow": {
                "operating_cash_flow": round(operating_cash_flow, 2),
                "depreciation": round(depreciation, 2),
                "capital_expenditures": round(capex, 2),
                "investing_cash_flow": round(investing_cash_flow, 2),
                "financing_cash_flow": round(financing_cash_flow, 2),
                "net_change_in_cash": round(
                    operating_cash_flow + investing_cash_flow + financing_cash_flow, 2
                ),
                "free_cash_flow": round(operating_cash_flow + capex, 2),
            },
        }
        
        financials["statements"].append(statement)
    
    return financials


def generate_legal_documents(company_id: str) -> dict[str, Any]:
    """Generate synthetic legal documents for a company."""
    
    company = COMPANIES[company_id]
    risk_profile = company["risk_profile"]
    
    # Number of items based on risk profile
    litigation_count = {
        "low": random.randint(0, 2),
        "medium": random.randint(2, 5),
        "high": random.randint(5, 10),
    }[risk_profile]
    
    contract_count = random.randint(10, 25)
    ip_count = random.randint(5, 20)
    
    legal_data = {
        "company_id": company_id,
        "company_name": company["name"],
        "litigations": [],
        "contracts": [],
        "ip_assets": [],
        "compliance": {},
    }
    
    # Generate litigations
    litigation_types = ["civil", "regulatory", "arbitration"]
    for i in range(litigation_count):
        amount = random.randint(100000, 5000000)
        legal_data["litigations"].append({
            "case_id": f"LIT-{company_id}-{2020+i}-{random.randint(100,999)}",
            "case_type": random.choice(litigation_types),
            "status": random.choice(["pending", "ongoing", "settled"]),
            "plaintiff": random.choice(["Former Employee", "Competitor", "Customer", "Regulatory Body"]),
            "defendant": company["name"],
            "amount_claimed": amount,
            "filing_date": f"{2020 + random.randint(0, 4)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "description": random.choice([
                "Employment discrimination claim",
                "Contract breach dispute",
                "Intellectual property infringement",
                "Regulatory compliance violation",
                "Product liability claim",
            ]),
        })
    
    # Generate contracts
    contract_types = [
        "Customer Agreement", "Supplier Contract", "Service Agreement",
        "License Agreement", "Partnership Agreement", "Lease Agreement",
    ]
    for i in range(contract_count):
        has_coc = random.random() < 0.3  # 30% have CoC provisions
        legal_data["contracts"].append({
            "contract_id": f"CON-{company_id}-{random.randint(1000, 9999)}",
            "contract_type": random.choice(contract_types),
            "counterparty": f"Company_{random.randint(1, 100)}",
            "value": random.randint(100000, 10000000),
            "start_date": f"{2020 + random.randint(0, 3)}-{random.randint(1,12):02d}-01",
            "end_date": f"{2025 + random.randint(0, 3)}-{random.randint(1,12):02d}-01",
            "has_change_of_control": has_coc,
            "coc_provision": "Termination right on CoC" if has_coc else None,
            "auto_renewal": random.random() < 0.5,
        })
    
    # Generate IP assets
    ip_types = ["patent", "trademark", "copyright"]
    for i in range(ip_count):
        legal_data["ip_assets"].append({
            "ip_id": f"IP-{company_id}-{random.randint(1000, 9999)}",
            "ip_type": random.choice(ip_types),
            "title": f"Innovation_{random.randint(1, 100)}",
            "status": random.choices(["granted", "pending", "expired"], weights=[0.7, 0.2, 0.1])[0],
            "filing_date": f"{2015 + random.randint(0, 8)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "expiry_date": f"{2030 + random.randint(0, 10)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "jurisdiction": random.choice(["India", "US", "EU", "Global"]),
        })
    
    # Generate compliance data
    legal_data["compliance"] = {
        "tax_filings_current": random.random() > 0.1,
        "regulatory_licenses": random.randint(3, 10),
        "licenses_valid": random.random() > 0.05,
        "data_privacy_compliant": random.random() > 0.2,
        "environmental_compliance": random.random() > 0.15,
        "last_audit_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "audit_result": random.choices(["clean", "minor_issues", "major_issues"], weights=[0.6, 0.3, 0.1])[0],
    }
    
    return legal_data


def generate_hr_data(company_id: str) -> dict[str, Any]:
    """Generate synthetic HR data for a company."""
    
    company = COMPANIES[company_id]
    employee_count = company["employees"]
    risk_profile = company["risk_profile"]
    
    # Attrition based on risk profile
    attrition_rates = {
        "low": random.uniform(0.08, 0.12),
        "medium": random.uniform(0.12, 0.18),
        "high": random.uniform(0.18, 0.25),
    }
    
    hr_data = {
        "company_id": company_id,
        "company_name": company["name"],
        "employee_metrics": {
            "total_employees": employee_count,
            "full_time": int(employee_count * 0.85),
            "part_time": int(employee_count * 0.05),
            "contractors": int(employee_count * 0.10),
            "average_tenure_years": random.uniform(2.5, 5.0),
            "median_salary": random.randint(50000, 150000),
        },
        "attrition": {
            "annual_rate": attrition_rates[risk_profile],
            "voluntary_percentage": random.uniform(0.6, 0.8),
            "key_departures_last_year": random.randint(0, 5),
            "industry_benchmark": 0.15,
        },
        "key_persons": [],
        "compliance": {},
        "culture": {},
        "policies": [],
    }
    
    # Generate key persons
    roles = [
        "CEO", "CFO", "CTO", "COO", "VP Engineering", 
        "VP Sales", "VP Marketing", "General Counsel"
    ]
    for role in roles:
        hr_data["key_persons"].append({
            "role": role,
            "tenure_years": random.uniform(1, 10),
            "has_succession_plan": random.random() > 0.4,
            "retention_package": random.random() > 0.5,
            "criticality": random.choice(["high", "critical"]) if "C" in role else "high",
        })
    
    # HR compliance
    hr_data["compliance"] = {
        "employment_disputes": random.randint(0, 3) if risk_profile == "low" else random.randint(2, 8),
        "discrimination_claims": random.randint(0, 2) if risk_profile != "high" else random.randint(1, 5),
        "wage_violations": random.randint(0, 2),
        "safety_incidents": random.randint(0, 5),
        "pending_investigations": 0 if risk_profile == "low" else random.randint(0, 2),
        "union_presence": random.random() < 0.2,
    }
    
    # Culture metrics
    hr_data["culture"] = {
        "employee_satisfaction": random.uniform(60, 90) if risk_profile != "high" else random.uniform(45, 70),
        "glassdoor_rating": random.uniform(3.2, 4.5) if risk_profile != "high" else random.uniform(2.5, 3.5),
        "work_style": random.choice(["hybrid", "remote", "office"]),
        "diversity_score": random.uniform(0.5, 0.9),
    }
    
    # HR Policies
    policies = [
        "Employee Handbook",
        "Code of Conduct",
        "Anti-Harassment Policy",
        "Remote Work Policy",
        "Data Privacy Policy",
        "Contractor Agreement Template",
        "NDA Template",
        "Non-Compete Agreement",
        "Benefits Guide",
        "Leave Policy",
    ]
    
    # Some companies might be missing policies
    available_policies = random.sample(policies, k=random.randint(7, 10))
    hr_data["policies"] = available_policies
    hr_data["missing_policies"] = [p for p in policies if p not in available_policies]
    
    return hr_data


def generate_all_synthetic_data(output_dir: str = "./data/synthetic"):
    """Generate all synthetic data and save to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "financial").mkdir(exist_ok=True)
    (output_path / "legal").mkdir(exist_ok=True)
    (output_path / "hr").mkdir(exist_ok=True)
    
    all_data = {
        "companies": COMPANIES,
        "financial": {},
        "legal": {},
        "hr": {},
    }
    
    for company_id in COMPANIES:
        print(f"Generating data for {company_id}...")
        
        # Generate financial data
        financial = generate_financial_statements(company_id)
        all_data["financial"][company_id] = financial
        with open(output_path / "financial" / f"{company_id}_financials.json", "w") as f:
            json.dump(financial, f, indent=2)
        
        # Generate legal data
        legal = generate_legal_documents(company_id)
        all_data["legal"][company_id] = legal
        with open(output_path / "legal" / f"{company_id}_legal.json", "w") as f:
            json.dump(legal, f, indent=2)
        
        # Generate HR data
        hr = generate_hr_data(company_id)
        all_data["hr"][company_id] = hr
        with open(output_path / "hr" / f"{company_id}_hr.json", "w") as f:
            json.dump(hr, f, indent=2)
    
    # Save combined data
    with open(output_path / "all_data.json", "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nSynthetic data generated successfully in {output_path}")
    print(f"Companies: {list(COMPANIES.keys())}")
    
    return all_data


class SyntheticDataGenerator:
    """Class wrapper for synthetic data generation functions."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the generator.
        
        Args:
            output_dir: Output directory for generated data. Defaults to data/generated.
        """
        if output_dir:
            self.output_path = Path(output_dir)
        else:
            self.output_path = Path(__file__).parent / "generated"
    
    def generate_company_data(
        self,
        company_id: str,
        company_name: str = None,
        industry: str = None,
    ) -> dict[str, Any]:
        """Generate all data for a company.
        
        Args:
            company_id: Company identifier
            company_name: Company name (optional, uses default if not provided)
            industry: Industry (optional)
            
        Returns:
            Dictionary with financial, legal, and HR data
        """
        # Ensure company exists in COMPANIES dict
        if company_id not in COMPANIES:
            # Add a custom company profile
            COMPANIES[company_id] = {
                "name": company_name or f"{company_id} Inc.",
                "industry": industry or "Technology",
                "founded": 2015,
                "headquarters": "Bangalore, India",
                "employees": 1000,
                "base_revenue": 100_000_000,
                "growth_rate": 0.10,
                "risk_profile": "medium",
            }
        
        result = {
            "company_id": company_id,
            "financial": generate_financial_statements(company_id),
            "legal": generate_legal_documents(company_id),
            "hr": generate_hr_data(company_id),
        }
        
        # Save to files
        company_path = self.output_path / company_id
        company_path.mkdir(parents=True, exist_ok=True)
        
        with open(company_path / "financial_data.json", "w") as f:
            json.dump(result["financial"], f, indent=2)
        
        with open(company_path / "legal_data.json", "w") as f:
            json.dump(result["legal"], f, indent=2)
        
        with open(company_path / "hr_data.json", "w") as f:
            json.dump(result["hr"], f, indent=2)
        
        return result
    
    def generate_all_companies(self) -> dict[str, Any]:
        """Generate data for all predefined companies.
        
        Returns:
            Dictionary with all company data
        """
        return generate_all_synthetic_data(str(self.output_path))
    
    @staticmethod
    def get_company_list() -> list[str]:
        """Get list of available company IDs."""
        return list(COMPANIES.keys())
    
    @staticmethod
    def get_company_profile(company_id: str) -> dict[str, Any]:
        """Get company profile."""
        return COMPANIES.get(company_id)


if __name__ == "__main__":
    generate_all_synthetic_data()
