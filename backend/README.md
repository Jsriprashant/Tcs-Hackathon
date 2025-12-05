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


### not useful
let me tell you how the legal agent will work
so this agent will have a md file, which has teh common legal problems faced by company like tcs, and which file to look for to find that problem.
the sample files are present in legal folder you can check them
Real contract and agreement folder as the dummy data for the merger company for testing.
all these document will be loaded into croma db in chunks and rag agent will fetch the relevant chunks.
now legal agent will follow the md file and then it finds out if thar problem exist in the dummy company data, if yes then it will flag it.

now if some company's data is not present then ask the user for the document and if user does not provide it then make a web search if that document is a public document, if no then flag it as it is a problem.

now after the legal agent finishes the md then it should compare the company document with the relevant chunks of data and find out more descrepancies which could hamper the future of the company which is merging the other company.

now the result of this legal agent will be based on certaon parameters, present in the knlodge markdown file, the parametrs are IP inrigment, ip descrepency, pending cases which can hamper company which is merging other company, descrepancy in contract duration maybe the contract ends just after merging and other parameters which can hamper the company.

then this result will be converted to json or any other structure which is feaseable, scalable

perform a root cause analysis and tell me how to implement this, i dont want any fallbacks and workarounds.
understand the codes throughly then create a markdown files in the location backend\docs\legal_agent{markdownfiles}.md
the implementaiton should be scalable, feaseable and efficient
without fallabckas and workarounds

if you have any query ask me before proceding.
dont worry about rag agent, it is being taken care of by another person.

Clarifying Questions
Before I create the comprehensive implementation plan for the Legal Agent, I need to understand a few things:

Knowledge Markdown File Location: You mentioned the legal agent will have an MD file with common legal problems. Should this knowledge base be located at:

backend/src/legal_agent/knowledge/ or
backend/data/legal_knowledge/ or
somewhere else?
Web Search Integration: For cases where documents are not present and user doesn't provide them, you mentioned web search for public documents. Which service should I use for web search?

Should I integrate with a search API (Tavily, SerpAPI, DuckDuckGo)?
Or should I assume an MCP server will provide this capability?
User Interaction for Missing Documents: When the agent needs to ask the user for documents:

Should this be done via the chat interface (interrupt the graph and wait for user upload)?
Should it use the human-in-the-loop pattern from LangGraph?
Output Structure: You mentioned the result should be converted to JSON or any scalable structure. Should the output:

Be a standalone report that gets saved to a file?
Be stored in the LangGraph state for downstream agents?
Be both (stored in state AND exported to a structured file)?
Company Data Mapping: The sample data has company1 and company2 folders. Should I:

Create a mapping to identify which company data belongs to which company ID (BBD, Supernova, etc.)?
Are company1 = Supernova and company2 = BBD based on the content I saw?
Parallel Execution: Should the legal checklist items be evaluated:

Sequentially (one after another)?
In parallel (for faster execution)?
In logical groups (e.g., all IP checks together, all contract checks together)?
Integration with Supervisor: The supervisor currently calls legal_agent as a sub-agent. Should the new implementation:

Keep the same interface (be called the same way)?
Change the calling pattern to support the new structured workflow?
Please clarify these points so I can create a precise, production-ready implementation plan without fallbacks or workarounds.

it will be present in backend\src\legal_agent\legal_knowledge
integrate a free search api
use human in the loop, in the front end i have a upload capability but it should this is for later development after the agent is ready, until then just mention the missing docs
store it in langgraph state, for downstream agents
if it helps in development then you can do it
preferablly in logical groups
we will update the supervisor but it should not break other agents, it should be a sub agent only but we can update it