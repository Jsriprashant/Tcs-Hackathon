# AI-Powered M&A Due Diligence Orchestrator

An intelligent multi-agent system for comprehensive M&A (Mergers & Acquisitions) due diligence analysis, built with LangGraph and LangChain for the TCS GenAI Hackathon.

## 🎯 Overview

This solution automates the complex process of M&A due diligence by orchestrating multiple specialized AI agents:

- **Supervisor Agent**: Central orchestrator that routes requests to appropriate specialist agents
- **Finance Agent**: Analyzes financial health, profitability, liquidity, solvency, and valuations
- **Legal Agent**: Reviews litigation risks, contracts, IP portfolios, and regulatory compliance
- **HR Agent**: Assesses organizational culture, attrition risks, key person dependencies, and policies
- **Analyst Agent**: Provides strategic analysis, synergy calculations, and deal recommendations
- **RAG Agent**: Retrieves relevant documents from ChromaDB vector stores

## 📁 Project Structure

```
backend/
├── langgraph.json              # LangGraph server configuration
├── pyproject.toml              # Python dependencies and project config
├── .env.example                # Environment variables template
├── Dockerfile                  # Container build
├── docker-compose.yml          # Local dev with dependencies
├── data/
│   ├── __init__.py
│   ├── synthetic_data_generator.py   # Generates sample company data
│   └── document_loader.py            # Loads data into ChromaDB
├── src/
│   ├── __init__.py
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py         # Enterprise settings with TCS GenAI config
│   │   └── llm_config.py       # LLM and embedding model configuration
│   ├── common/                 # Shared utilities across all agents
│   │   ├── __init__.py
│   │   ├── state.py            # Shared state definitions (CompanyInfo, RiskScore, etc.)
│   │   ├── errors.py           # Custom exceptions
│   │   ├── logging_config.py   # Structured logging with structlog
│   │   ├── guardrails.py       # Security (PII filter, input validation)
│   │   └── utils.py            # Utility functions (formatting, calculations)
│   ├── supervisor/             # Supervisor Agent (orchestration)
│   │   ├── __init__.py
│   │   ├── graph.py            # Main orchestration graph
│   │   ├── prompts.py          # Supervisor prompts
│   │   └── state.py            # Supervisor state schema
│   ├── rag_agent/              # RAG Agent (document retrieval)
│   │   ├── __init__.py
│   │   ├── graph.py            # RAG agent graph
│   │   ├── tools.py            # Document retrieval tools
│   │   └── state.py            # RAG agent state
│   ├── finance_agent/          # Finance Agent (financial analysis)
│   │   ├── __init__.py
│   │   ├── graph.py            # Finance agent graph
│   │   ├── tools.py            # Financial analysis tools
│   │   └── state.py            # Finance agent state
│   ├── legal_agent/            # Legal Agent (legal due diligence)
│   │   ├── __init__.py
│   │   ├── graph.py            # Legal agent graph
│   │   ├── tools.py            # Legal analysis tools
│   │   └── state.py            # Legal agent state
│   ├── hr_agent/               # HR Agent (people & culture)
│   │   ├── __init__.py
│   │   ├── graph.py            # HR agent graph
│   │   ├── tools.py            # HR analysis tools
│   │   └── state.py            # HR agent state
│   └── analyst_agent/          # Analyst Agent (strategic analysis)
│       ├── __init__.py
│       ├── graph.py            # Analyst agent graph
│       ├── tools.py            # Strategic analysis tools
│       └── state.py            # Analyst agent state
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   └── test_*.py               # Test files
└── scripts/
    ├── run_demo.py             # Demo script
    └── setup_data.py           # Data initialization script
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- TCS GenAI Lab API access (or OpenAI API key)
- pip or uv package manager

### 2. Setup Environment

```bash
# Navigate to backend directory
cd backend

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required variables:
#   TCS_GENAI_API_KEY=your-api-key
#   TCS_GENAI_BASE_URL=https://genai-api.tcs.com/v1  (or your endpoint)
```

### 3. Install Dependencies

```bash
# Using pip
pip install -e .

# Or using uv (faster)
uv pip install -e .
```

### 4. Initialize Data

```bash
# Generate synthetic company data and load into ChromaDB
python -c "from data.synthetic_data_generator import generate_all_data; generate_all_data()"
python -c "from data.document_loader import load_all_documents; load_all_documents()"
```

### 5. Run the Server

```bash
# Start LangGraph development server
langgraph dev --allow-blocking

# Or with Docker (includes PostgreSQL for checkpointing)
docker-compose up -d
```

### 6. Test the System

```bash
# Run the demo script
python scripts/run_demo.py

# Or use the LangChain Chat UI
# Navigate to http://localhost:8000 in your browser
```

## 🏢 Sample Companies

The system includes synthetic data for 5 companies across different industries:

| Company | Industry | Profile |
|---------|----------|---------|
| **TECHCORP** | Technology | High-growth SaaS company with strong margins |
| **FINSERV** | Financial Services | Established financial services with regulatory exposure |
| **HEALTHTECH** | Healthcare Technology | Healthcare IT company with IP portfolio |
| **RETAILMAX** | Retail | Multi-channel retailer with supply chain complexity |
| **GREENERGY** | Clean Energy | Renewable energy with government contracts |

## 📊 Analysis Domains

### Financial Analysis
- **Profitability**: Revenue trends, margins, EBITDA analysis
- **Liquidity**: Current ratio, quick ratio, working capital
- **Solvency**: Debt-to-equity, interest coverage, leverage
- **Cash Flow**: Operating cash flow, free cash flow, burn rate
- **Valuation**: DCF, comparable company analysis, precedent transactions

### Legal Analysis
- **Litigation**: Pending cases, settlement history, exposure assessment
- **Contracts**: Material agreements, change of control provisions
- **IP Portfolio**: Patents, trademarks, trade secrets
- **Compliance**: Regulatory status, violations, remediation plans

### HR Analysis
- **Attrition**: Turnover rates, department analysis, retention programs
- **Key Persons**: Executive dependencies, succession planning
- **Culture**: Employee satisfaction, cultural compatibility
- **Policies**: Compensation, benefits, employment agreements

### Strategic Analysis
- **Merger Type**: Horizontal vs vertical integration assessment
- **Synergies**: Revenue and cost synergy calculations
- **Deal Recommendation**: Go/No-Go with confidence scoring

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TCS_GENAI_API_KEY` | TCS GenAI Lab API key | - |
| `TCS_GENAI_BASE_URL` | TCS GenAI Lab base URL | `https://genai-api.tcs.com/v1` |
| `LLM_MODEL` | Model name | `gpt-4o` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./data/chroma_db` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DATABASE_URL` | PostgreSQL URL for checkpointing | - |

### Risk Thresholds

Configure risk scoring thresholds in `src/config/settings.py`:

```python
RISK_THRESHOLDS = {
    "financial": {"low": 0.3, "medium": 0.6, "high": 0.8},
    "legal": {"low": 0.25, "medium": 0.5, "high": 0.75},
    "hr": {"low": 0.2, "medium": 0.5, "high": 0.7},
    "overall": {"low": 0.3, "medium": 0.55, "high": 0.75}
}
```

## 🔒 Security Features

- **PII Filtering**: Automatic detection and redaction of sensitive information
- **Input Validation**: Sanitization of user inputs before processing
- **Output Sanitization**: Removal of potentially harmful content
- **Content Moderation**: Guardrails for appropriate responses
- **Audit Logging**: Structured logs for compliance and debugging

## 📈 Risk Scoring Framework

Each analysis produces a normalized risk score (0-1):

- **0.0 - 0.3**: Low Risk (Green) ✅
- **0.3 - 0.6**: Medium Risk (Yellow) ⚠️
- **0.6 - 0.8**: High Risk (Orange) 🔶
- **0.8 - 1.0**: Critical Risk (Red) 🔴

Overall deal risk is calculated as a weighted average:
- Financial: 35%
- Legal: 30%
- HR: 15%
- Strategic: 20%

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific agent tests
pytest tests/test_finance_agent.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t dd-orchestrator .

# Run with docker-compose (includes PostgreSQL)
docker-compose up -d

# Check logs
docker-compose logs -f backend
```

## 📚 API Usage

### Using LangGraph Client

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:8000")

# Start a due diligence analysis
result = client.runs.create(
    assistant_id="dd-supervisor",
    input={
        "messages": [
            {"role": "user", "content": "Analyze TECHCORP for potential acquisition"}
        ]
    }
)

# Stream the response
for chunk in client.runs.stream(result["run_id"]):
    print(chunk)
```

### Example Queries

```
# Financial Analysis
"What is the financial health of TECHCORP?"
"Analyze profitability trends for FINSERV over the last 3 years"

# Legal Analysis
"What are the litigation risks for HEALTHTECH?"
"Review material contracts for RETAILMAX"

# HR Analysis
"Assess the key person risk for GREENERGY"
"What is the attrition rate at TECHCORP?"

# Strategic Analysis
"Should we proceed with acquiring FINSERV? Provide a deal recommendation"
"Calculate potential synergies for a TECHCORP-HEALTHTECH merger"
```

## 🛠️ Development

### Adding a New Agent

1. Create new directory: `src/your_agent/`
2. Implement required files:
   - `__init__.py`: Module exports
   - `state.py`: Agent state schema
   - `tools.py`: Agent-specific tools with `@tool` decorator
   - `graph.py`: LangGraph StateGraph definition
3. Register in `langgraph.json`
4. Add routing logic in `src/supervisor/graph.py`

### Adding New Tools

```python
from langchain_core.tools import tool

@tool
def analyze_something(company_id: str, parameter: str) -> dict:
    """Analyze something specific for a company.
    
    Args:
        company_id: The company identifier
        parameter: Analysis parameter
        
    Returns:
        Analysis results dictionary
    """
    # Implementation
    return {"result": "analysis"}
```

## 📄 License

TCS Internal Use Only - Hackathon Project

## 👥 Team

TCS GenAI Hackathon Team

## 🙏 Acknowledgments

- LangChain and LangGraph teams for the excellent frameworks
- TCS GenAI Lab for API access and support
