"""State definitions for RAG Agent."""

from typing import Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from src.common.state import BaseAgentState


class DocumentMetadata(BaseModel):
    """Metadata for a document in the RAG system."""
    document_id: str = ""
    company_id: str = ""
    document_type: Literal[
        "income_statement", 
        "balance_sheet", 
        "cash_flow",
        "financial_statement",
        "contract",
        "litigation",
        "ip_document",
        "compliance",
        "hr_policy",
        "employee_record"
    ] = "financial_statement"
    category: Literal["financial", "legal", "hr", "market"] = "financial"
    fiscal_year: Optional[int] = None
    upload_date: Optional[str] = None
    source: Optional[str] = None


class RetrievalRequest(BaseModel):
    """Request for document retrieval."""
    query: str
    company_ids: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5)
    min_similarity: float = Field(default=0.7)


class RetrievedDocument(BaseModel):
    """A retrieved document with relevance score."""
    content: str
    metadata: DocumentMetadata
    similarity_score: float = 0.0
    chunk_id: Optional[str] = None


class RAGAgentState(BaseAgentState):
    """State for RAG Agent operations."""
    
    # Messages
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    
    # Deal context
    deal_type: Literal["merger", "acquisition"] = "acquisition"
    target_company: Optional[dict] = None
    
    # Retrieval context
    retrieval_request: Optional[RetrievalRequest] = None
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    
    # Collection management
    active_collections: list[str] = Field(default_factory=list)
    
    # Context for agents
    context_for_finance: Optional[str] = None
    context_for_legal: Optional[str] = None
    context_for_hr: Optional[str] = None
    context_for_analyst: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
