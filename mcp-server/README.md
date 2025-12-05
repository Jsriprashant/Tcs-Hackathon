# MCP Server Skeleton

A reusable Model Context Protocol (MCP) server skeleton using FastMCP. This provides a production-ready template for building MCP servers that integrate with LangGraph agents.

## Features

- 🚀 **FastMCP Framework**: Modern, async-first MCP server implementation
- 🔧 **Modular Tool Architecture**: Easily add, remove, or modify tools
- 🔒 **Configuration Management**: Pydantic settings with environment variables
- 📝 **Structured Logging**: Consistent logging with structlog
- 🐳 **Docker Support**: Ready for containerized deployment
- ✅ **Testing Ready**: pytest setup with fixtures and examples

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
vim .env
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"
```

### 3. Run the Server

```bash
# Development mode with hot reload
python -m src.server

# Or using the MCP CLI
mcp dev src/server.py

# Or run via script
./scripts/run_dev.sh
```

### 4. Test the Server

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Smoke test
python scripts/run_smoke_test.py
```

## Project Structure

```
mcp-server-skeleton/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── server.py                # Main MCP server entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Pydantic settings configuration
│   ├── tools/
│   │   ├── __init__.py          # Tool registration
│   │   ├── base.py              # Base tool utilities
│   │   ├── example_tools.py     # Example tool implementations
│   │   └── your_tools.py        # Add your custom tools here
│   └── common/
│       ├── __init__.py
│       ├── logging_config.py    # Logging configuration
│       └── errors.py            # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_server.py           # Server tests
│   └── test_tools.py            # Tool tests
├── config/
│   └── tool_config.yaml         # Tool-specific configuration
├── scripts/
│   ├── run_dev.sh              # Development server script
│   └── run_smoke_test.py       # Smoke test script
├── .env.example                 # Environment template
├── Dockerfile                   # Container build
├── docker-compose.yml           # Local development
├── pyproject.toml              # Python project config
└── README.md                   # This file
```

## Adding New Tools

### 1. Create a New Tool File

Create a new file in `src/tools/`:

```python
# src/tools/my_tools.py
from mcp.server.fastmcp import FastMCP
from src.common.logging_config import get_logger

logger = get_logger(__name__)

def register_tools(mcp: FastMCP):
    """Register tools with the MCP server."""
    
    @mcp.tool()
    def my_custom_tool(
        param1: str,
        param2: int = 10,
    ) -> str:
        """
        Description of what this tool does.
        
        Args:
            param1: Description of param1
            param2: Description of param2 (default: 10)
        
        Returns:
            Description of return value
        """
        logger.info("my_custom_tool called", param1=param1, param2=param2)
        
        # Your tool logic here
        result = f"Processed {param1} with {param2}"
        
        return result
```

### 2. Register in `__init__.py`

Update `src/tools/__init__.py`:

```python
from .example_tools import register_tools as register_example_tools
from .my_tools import register_tools as register_my_tools

def register_all_tools(mcp):
    """Register all tools with the MCP server."""
    register_example_tools(mcp)
    register_my_tools(mcp)
```

### 3. Tool Best Practices

- **Clear Names**: Use descriptive, action-oriented names (e.g., `create_report`, `search_documents`)
- **Type Hints**: Always include type hints for parameters and return values
- **Docstrings**: Write clear descriptions - these become tool documentation
- **Error Handling**: Return user-friendly error messages
- **Logging**: Log tool invocations for debugging
- **Validation**: Validate inputs early and fail fast

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_SERVER_HOST` | Server host | `0.0.0.0` |
| `MCP_SERVER_PORT` | Server port | `8000` |
| `MCP_SERVER_NAME` | Server name | `mcp-server` |
| `ENVIRONMENT` | Environment name | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format (`json`/`console`) | `console` |

### Tool Configuration

Configure tool-specific settings in `config/tool_config.yaml`:

```yaml
tools:
  example_tool:
    enabled: true
    max_retries: 3
    timeout: 30
  
  another_tool:
    enabled: true
    api_key: "${ANOTHER_TOOL_API_KEY}"
```

## Integration with LangGraph Backend

This MCP server is designed to work with the `backend-skeleton`. The backend's `MCPClient` connects to this server via HTTP.

### Backend Configuration

In the backend's `.env`:

```env
MCP_SERVER_URL=http://localhost:8000/mcp
```

### Testing Integration

1. Start the MCP server:
   ```bash
   cd mcp-server-skeleton
   ./scripts/run_dev.sh
   ```

2. Start the backend:
   ```bash
   cd backend-skeleton
   ./scripts/run_dev.sh
   ```

3. The backend will automatically discover and use tools from the MCP server.

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t mcp-server .

# Run the container
docker run -p 8000:8000 --env-file .env mcp-server
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f mcp-server

# Stop services
docker-compose down
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Test Specific Tools

```bash
pytest tests/test_tools.py -k "test_example"
```

## Troubleshooting

### Server Won't Start

1. Check if port is already in use:
   ```bash
   lsof -i :8000
   ```

2. Check environment variables:
   ```bash
   python -c "from src.config.settings import get_settings; print(get_settings())"
   ```

### Tools Not Found

1. Verify tool registration in `src/tools/__init__.py`
2. Check server logs for registration errors
3. Test tool discovery:
   ```bash
   curl http://localhost:8000/mcp/tools
   ```

### Integration Issues

1. Verify MCP server is running and accessible
2. Check backend's `MCP_SERVER_URL` configuration
3. Test connectivity:
   ```bash
   curl http://localhost:8000/health
   ```

## License

MIT License - feel free to use this skeleton for your projects.
