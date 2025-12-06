"""
FastAPI application for RAG queries.

Provides REST API endpoint for document retrieval and generation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time

from src.rag_agent.retrieve import get_retriever
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="M&A Due Diligence RAG API",
    description="RAG-powered document retrieval and Q&A for M&A due diligence",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RAGQueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="User query")
    category: Optional[str] = Field(None, description="Document category filter (financial, legal, hr, market)")
    company_id: Optional[str] = Field(None, description="Company ID filter")
    doc_type: Optional[str] = Field(None, description="Document type filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of context documents to retrieve")
    include_generation: bool = Field(True, description="Generate answer from contexts")


class RAGQueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    latency: float
    num_sources: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "M&A Due Diligence RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/rag_query": "POST - Query documents and get AI-generated answers",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/rag_query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Query RAG system for relevant documents and optionally generate an answer.
    
    Args:
        request: RAGQueryRequest with query and filters
        
    Returns:
        RAGQueryResponse with contexts, answer, and metadata
    """
    try:
        logger.info(f"RAG query: {request.query[:100]}...")
        
        # Get retriever
        retriever = get_retriever()
        
        # Execute query
        result = retriever.query(
            query=request.query,
            category=request.category,
            company_id=request.company_id,
            top_k=request.top_k,
            include_generation=request.include_generation,
        )
        
        # Build response
        response = RAGQueryResponse(
            query=request.query,
            answer=result.get("answer"),
            sources=result.get("contexts", []),
            latency=result.get("latency", 0.0),
            num_sources=len(result.get("contexts", []))
        )
        
        logger.info(f"Query completed in {response.latency:.3f}s with {response.num_sources} sources")
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def get_categories():
    """Get available document categories."""
    return {
        "categories": [
            {"value": "financial", "label": "Financial"},
            {"value": "legal", "label": "Legal"},
            {"value": "hr", "label": "HR"},
            {"value": "market", "label": "Market"},
        ]
    }


@app.get("/companies")
async def get_companies():
    """Get available companies."""
    return {
        "companies": [
            {"id": "BBD", "name": "BBD Ltd"},
            {"id": "XYZ", "name": "XYZ Ltd"},
            {"id": "SUPERNOVA", "name": "Supernova Inc"},
            {"id": "RASPUTIN", "name": "Rasputin Petroleum Ltd"},
            {"id": "TECHNOBOX", "name": "Techno Box Inc"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
