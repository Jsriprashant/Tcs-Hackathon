#!/bin/bash
# run_dev.sh - Development server startup script

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
echo -e "${GREEN}  Backend Skeleton Development Server${NC}"
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
PORT="${LANGGRAPH_API_PORT:-8123}"
HOST="${LANGGRAPH_API_HOST:-0.0.0.0}"
RELOAD="--reload"

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
        --no-reload)
            RELOAD=""
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT      Port to run on (default: 8123)"
            echo "  --host HOST      Host to bind to (default: 0.0.0.0)"
            echo "  --no-reload      Disable auto-reload"
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
echo ""

# Check if langgraph CLI is available
if command -v langgraph &> /dev/null; then
    echo -e "${GREEN}Starting LangGraph development server...${NC}"
    echo ""
    
    # Run with LangGraph CLI
    langgraph dev \
        --host "$HOST" \
        --port "$PORT" \
        $RELOAD
else
    echo -e "${YELLOW}LangGraph CLI not found, falling back to uvicorn...${NC}"
    echo ""
    
    # Fallback to uvicorn
    python -m uvicorn src.api:app \
        --host "$HOST" \
        --port "$PORT" \
        $RELOAD
fi
