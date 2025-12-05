"""Tests for the Due Diligence Orchestrator agents."""

import pytest
from unittest.mock import MagicMock, patch

# Test imports
def test_config_imports():
    """Test that configuration modules can be imported."""
    from src.config.settings import Settings, get_settings
    from src.config.llm_config import get_llm, get_embedding_model
    
    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, 'app_name')


def test_common_imports():
    """Test that common modules can be imported."""
    from src.common.state import CompanyInfo, RiskScore, AnalysisResult
    from src.common.errors import DueDiligenceError, AgentError
    from src.common.utils import format_currency, calculate_risk_score
    from src.common.guardrails import PIIFilter, InputValidator
    
    # Test basic functionality
    assert format_currency(1000000) == "$1,000,000.00"
    
    # Test CompanyInfo creation
    company = CompanyInfo(
        company_id="TEST001",
        company_name="Test Company",
        industry="Technology"
    )
    assert company.company_id == "TEST001"


def test_risk_score_calculation():
    """Test risk score calculation."""
    from src.common.utils import calculate_risk_score
    
    # Test with sample metrics
    scores = {
        "financial_health": 0.7,
        "legal_exposure": 0.3,
        "operational_risk": 0.5
    }
    weights = {
        "financial_health": 0.4,
        "legal_exposure": 0.35,
        "operational_risk": 0.25
    }
    
    overall = calculate_risk_score(scores, weights)
    assert 0 <= overall <= 1


def test_pii_filter():
    """Test PII filtering."""
    from src.common.guardrails import PIIFilter
    
    filter = PIIFilter()
    
    # Test email filtering
    text_with_email = "Contact john.doe@example.com for info"
    filtered = filter.filter(text_with_email)
    assert "john.doe@example.com" not in filtered
    
    # Test phone filtering
    text_with_phone = "Call 123-456-7890 for support"
    filtered = filter.filter(text_with_phone)
    assert "123-456-7890" not in filtered


def test_input_validator():
    """Test input validation."""
    from src.common.guardrails import InputValidator
    
    validator = InputValidator()
    
    # Test valid input
    valid_input = "Analyze the financial health of TechCorp"
    is_valid, _ = validator.validate(valid_input)
    assert is_valid
    
    # Test empty input
    is_valid, error = validator.validate("")
    assert not is_valid


def test_agent_state_imports():
    """Test that agent state modules can be imported."""
    from src.finance_agent.state import FinanceAgentState
    from src.legal_agent.state import LegalAgentState
    from src.hr_agent.state import HRAgentState
    from src.analyst_agent.state import AnalystAgentState
    from src.rag_agent.state import RAGAgentState
    from src.supervisor.state import SupervisorState
    
    # All imports should succeed
    assert FinanceAgentState is not None
    assert LegalAgentState is not None
    assert HRAgentState is not None
    assert AnalystAgentState is not None
    assert RAGAgentState is not None
    assert SupervisorState is not None


def test_synthetic_data_generator():
    """Test synthetic data generator."""
    from data.synthetic_data_generator import (
        SyntheticDataGenerator,
        COMPANIES
    )
    
    # Test company list
    assert "TECHCORP" in COMPANIES
    assert "FINSERV" in COMPANIES
    assert len(COMPANIES) >= 5
    
    # Test generator initialization
    generator = SyntheticDataGenerator()
    assert generator is not None
    
    # Test company list method
    companies = generator.get_company_list()
    assert "TECHCORP" in companies


def test_financial_statement_generation():
    """Test financial statement generation."""
    from data.synthetic_data_generator import generate_financial_statements
    
    financials = generate_financial_statements("TECHCORP")
    
    assert "company_id" in financials
    assert financials["company_id"] == "TECHCORP"
    assert "statements" in financials
    assert len(financials["statements"]) > 0
    
    # Check statement structure
    statement = financials["statements"][0]
    assert "fiscal_year" in statement
    assert "income_statement" in statement
    assert "balance_sheet" in statement
    assert "cash_flow" in statement


def test_legal_document_generation():
    """Test legal document generation."""
    from data.synthetic_data_generator import generate_legal_documents
    
    legal = generate_legal_documents("TECHCORP")
    
    assert "company_id" in legal
    assert legal["company_id"] == "TECHCORP"
    assert "litigations" in legal
    assert "contracts" in legal
    assert "intellectual_property" in legal


def test_hr_data_generation():
    """Test HR data generation."""
    from data.synthetic_data_generator import generate_hr_data
    
    hr = generate_hr_data("TECHCORP")
    
    assert "company_id" in hr
    assert hr["company_id"] == "TECHCORP"
    assert "employee_summary" in hr
    assert "key_executives" in hr


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB for testing without actual vector store."""
    with patch('data.document_loader.Chroma') as mock_chroma:
        mock_instance = MagicMock()
        mock_instance.add_documents = MagicMock()
        mock_instance.similarity_search = MagicMock(return_value=[])
        mock_chroma.return_value = mock_instance
        yield mock_chroma


def test_document_loader_init(mock_chromadb):
    """Test document loader initialization."""
    from data.document_loader import DocumentLoader, COLLECTIONS
    
    loader = DocumentLoader()
    assert loader is not None
    
    # Check collections are defined
    assert "financial" in COLLECTIONS
    assert "legal" in COLLECTIONS
    assert "hr" in COLLECTIONS


class TestFinanceAgent:
    """Tests for Finance Agent."""
    
    def test_state_creation(self):
        """Test finance agent state creation."""
        from src.finance_agent.state import FinanceAgentState
        from langchain_core.messages import HumanMessage
        
        state = FinanceAgentState(
            messages=[HumanMessage(content="Analyze profitability")],
            company_id="TECHCORP"
        )
        assert state.company_id == "TECHCORP"
    
    def test_tools_import(self):
        """Test finance agent tools can be imported."""
        from src.finance_agent.tools import (
            analyze_profitability,
            analyze_liquidity,
            analyze_solvency,
            analyze_cash_flow,
            analyze_valuation
        )
        
        assert analyze_profitability is not None
        assert analyze_liquidity is not None


class TestLegalAgent:
    """Tests for Legal Agent."""
    
    def test_state_creation(self):
        """Test legal agent state creation."""
        from src.legal_agent.state import LegalAgentState
        from langchain_core.messages import HumanMessage
        
        state = LegalAgentState(
            messages=[HumanMessage(content="Review contracts")],
            company_id="TECHCORP"
        )
        assert state.company_id == "TECHCORP"
    
    def test_tools_import(self):
        """Test legal agent tools can be imported."""
        from src.legal_agent.tools import (
            analyze_litigation_risk,
            analyze_contracts,
            analyze_ip_portfolio,
            analyze_regulatory_compliance
        )
        
        assert analyze_litigation_risk is not None


class TestHRAgent:
    """Tests for HR Agent."""
    
    def test_state_creation(self):
        """Test HR agent state creation."""
        from src.hr_agent.state import HRAgentState
        from langchain_core.messages import HumanMessage
        
        state = HRAgentState(
            messages=[HumanMessage(content="Analyze attrition")],
            company_id="TECHCORP"
        )
        assert state.company_id == "TECHCORP"
    
    def test_tools_import(self):
        """Test HR agent tools can be imported."""
        from src.hr_agent.tools import (
            analyze_attrition,
            analyze_key_persons,
            analyze_culture,
            analyze_hr_policies
        )
        
        assert analyze_attrition is not None


class TestAnalystAgent:
    """Tests for Analyst Agent."""
    
    def test_state_creation(self):
        """Test analyst agent state creation."""
        from src.analyst_agent.state import AnalystAgentState
        from langchain_core.messages import HumanMessage
        
        state = AnalystAgentState(
            messages=[HumanMessage(content="Recommend deal")],
            target_company_id="TECHCORP"
        )
        assert state.target_company_id == "TECHCORP"
    
    def test_tools_import(self):
        """Test analyst agent tools can be imported."""
        from src.analyst_agent.tools import (
            analyze_horizontal_merger,
            analyze_vertical_merger,
            calculate_synergies,
            generate_deal_recommendation
        )
        
        assert generate_deal_recommendation is not None


class TestSupervisor:
    """Tests for Supervisor Agent."""
    
    def test_state_creation(self):
        """Test supervisor state creation."""
        from src.supervisor.state import SupervisorState
        from langchain_core.messages import HumanMessage
        
        state = SupervisorState(
            messages=[HumanMessage(content="Analyze TechCorp")]
        )
        assert len(state.messages) == 1
    
    def test_prompts_import(self):
        """Test supervisor prompts can be imported."""
        from src.supervisor.prompts import (
            SUPERVISOR_SYSTEM_PROMPT,
            ROUTING_PROMPT
        )
        
        assert SUPERVISOR_SYSTEM_PROMPT is not None
        assert len(SUPERVISOR_SYSTEM_PROMPT) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
