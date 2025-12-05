"""
Base abstractions for RAG system.

Provides abstract base classes and common interfaces for ingestion and retrieval.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Literal
from pathlib import Path
from datetime import datetime
from enum import Enum


class DocumentCategory(str, Enum):
    """Document categories for M&A due diligence."""
    FINANCIAL = "financial"
    LEGAL = "legal"
    HR = "hr"
    MARKET = "market"
    CROSS_REF = "cross_ref"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    """Document types within categories."""
    # Financial
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    FINANCIAL_STATEMENT = "financial_statement"
    
    # Legal
    CONTRACT = "contract"
    LITIGATION = "litigation"
    IP_DOCUMENT = "ip_document"
    COMPLIANCE = "compliance"
    COURT_JUDGMENT = "court_judgment"
    NDA = "nda"
    LICENSE_AGREEMENT = "license_agreement"
    PARTNERSHIP_AGREEMENT = "partnership_agreement"
    ENVIRONMENTAL_POLICY = "environmental_policy"
    
    # HR
    EMPLOYEE_RECORD = "employee_record"
    HR_POLICY = "hr_policy"
    EMPLOYEE_HANDBOOK = "employee_handbook"
    POLICY_DOCUMENT = "policy_document"
    
    # Generic
    UNKNOWN = "unknown"


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    source: str
    filename: str
    company_id: str = "UNKNOWN"
    category: DocumentCategory = DocumentCategory.UNKNOWN
    doc_type: DocumentType = DocumentType.UNKNOWN
    chunk_hash: str = ""
    page: Optional[int] = None
    fiscal_year: Optional[int] = None
    upload_date: str = field(default_factory=lambda: datetime.now().isoformat())
    record_count: Optional[int] = None
    linked_to: List[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source": self.source,
            "filename": self.filename,
            "company_id": self.company_id,
            "category": self.category.value if isinstance(self.category, DocumentCategory) else self.category,
            "doc_type": self.doc_type.value if isinstance(self.doc_type, DocumentType) else self.doc_type,
            "chunk_hash": self.chunk_hash,
            "page": self.page,
            "fiscal_year": self.fiscal_year,
            "upload_date": self.upload_date,
            "record_count": self.record_count,
            "linked_to": ",".join(self.linked_to) if self.linked_to else "",
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


@dataclass
class DocumentChunk:
    """A chunk of document content with metadata."""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    content: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str = "semantic"  # semantic, bm25, hybrid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "retrieval_method": self.retrieval_method,
        }


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""
    files_processed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    chunks_deduplicated: int = 0
    total_characters: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def dedup_ratio(self) -> float:
        """Get deduplication ratio."""
        total = self.chunks_created + self.chunks_deduplicated
        if total == 0:
            return 0.0
        return self.chunks_deduplicated / total


class BaseRAG(ABC):
    """Abstract base class for RAG implementations."""
    
    @abstractmethod
    def ingest(self, path: Path, category: Optional[DocumentCategory] = None) -> IngestionStats:
        """
        Ingest documents from a path into the vector store.
        
        Args:
            path: Path to file or directory to ingest
            category: Optional category override
            
        Returns:
            IngestionStats with ingestion metrics
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        category: Optional[DocumentCategory] = None,
        company_id: Optional[str] = None,
        doc_type: Optional[DocumentType] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            category: Optional category filter
            company_id: Optional company filter
            doc_type: Optional document type filter
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved context documents
            system_prompt: Optional system prompt override
            
        Returns:
            Generated response string
        """
        pass


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: Path) -> List[DocumentChunk]:
        """
        Load a document and return chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of DocumentChunk objects
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """
        Check if this loader supports the given file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if this loader can handle the file
        """
        pass


class BaseChunker(ABC):
    """Abstract base class for text chunking."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata for context-aware chunking
            
        Returns:
            List of text chunks
        """
        pass


class BaseDeduplicator(ABC):
    """Abstract base class for deduplication."""
    
    @abstractmethod
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a duplicate
        """
        pass
    
    @abstractmethod
    def add(self, text: str) -> str:
        """
        Add text to dedup index and return hash.
        
        Args:
            text: Text to add
            
        Returns:
            Hash of the text
        """
        pass
    
    @abstractmethod
    def clear(self):
        """Clear the deduplication index."""
        pass
