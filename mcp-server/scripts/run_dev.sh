#!/bin/bash
# run_dev.sh - MCP Server development startup script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MCP Server Skeleton Development${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Copying .env.example to .env..."
    cp .env.example .env
    echo -e "${YELLOW}Please update .env with your configuration${NC}"
    echo ""
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -e ".[dev]"
    touch .venv/.deps_installed
fi

# Parse command line arguments
PORT="${MCP_SERVER_PORT:-8000}"
HOST="${MCP_SERVER_HOST:-0.0.0.0}"
MODE="server"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --dev)
            MODE="dev"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT      Port to run on (default: 8000)"
            echo "  --host HOST      Host to bind to (default: 0.0.0.0)"
            echo "  --dev            Run with MCP dev CLI (hot reload)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Environment: ${ENVIRONMENT:-development}"
echo "  Transport: ${MCP_TRANSPORT:-streamable-http}"
echo ""

if [ "$MODE" = "dev" ]; then
    # Check if MCP CLI is available
    if command -v mcp &> /dev/null; then
        echo -e "${GREEN}Starting with MCP dev CLI (hot reload)...${NC}"
        mcp dev src/server.py
    else
        echo -e "${YELLOW}MCP CLI not found, falling back to direct run...${NC}"
        python -m src.server
    fi
else
    echo -e "${GREEN}Starting MCP server...${NC}"
    python -m src.server
fi
