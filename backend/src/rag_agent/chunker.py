"""
Text chunking utilities for RAG ingestion.

Implements semantic and recursive character-based chunking.
"""

from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag_agent.base import BaseChunker
from src.common.logging_config import get_logger

logger = get_logger(__name__)


class SemanticChunker(BaseChunker):
    """Semantic chunker using sentence boundaries."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators for splitting (sentence-aware)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Sentence-aware separators
        if separators is None:
            separators = [
                "\n\n\n",  # Paragraph breaks
                "\n\n",    # Double newline
                "\n",      # Single newline
                ". ",      # Sentence end
                "! ",      # Exclamation
                "? ",      # Question
                "; ",      # Semicolon
                ": ",      # Colon
                ", ",      # Comma
                " ",       # Space
                ""         # Character-level fallback
            ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split text into semantic chunks."""
        if not text or not text.strip():
            return []
        
        try:
            chunks = self.splitter.split_text(text)
            # Filter out very small chunks
            chunks = [c for c in chunks if len(c.strip()) > 10]
            return chunks
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            # Fallback to simple splitting
            return self._simple_split(text)
    
    def _simple_split(self, text: str) -> List[str]:
        """Simple fallback splitting."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class LlamaIndexSemanticChunker(BaseChunker):
    """Semantic chunker using LlamaIndex SentenceSplitter."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize LlamaIndex semantic chunker.
        
        Args:
            chunk_size: Target size of chunks in tokens/characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Try to import LlamaIndex
        try:
            from llama_index.core.node_parser import SentenceSplitter
            self.splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.use_llama = True
            logger.info("Using LlamaIndex SentenceSplitter for chunking")
        except ImportError:
            logger.warning("LlamaIndex not available, falling back to RecursiveCharacterTextSplitter")
            self.use_llama = False
            self.fallback = SemanticChunker(chunk_size, chunk_overlap)
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split text using LlamaIndex or fallback."""
        if not text or not text.strip():
            return []
        
        if not self.use_llama:
            return self.fallback.chunk(text, metadata)
        
        try:
            # LlamaIndex expects list of text
            chunks = self.splitter.split_text(text)
            # Filter out very small chunks
            chunks = [c for c in chunks if len(c.strip()) > 10]
            return chunks
        except Exception as e:
            logger.error(f"Error in LlamaIndex chunking: {e}")
            # Fallback
            if hasattr(self, 'fallback'):
                return self.fallback.chunk(text, metadata)
            return []


class AdaptiveChunker(BaseChunker):
    """Adaptive chunker that selects strategy based on content."""
    
    def __init__(
        self,
        default_chunk_size: int = 512,
        default_overlap: int = 50
    ):
        """Initialize adaptive chunker."""
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        
        # Different chunkers for different content types
        self.semantic = LlamaIndexSemanticChunker(
            chunk_size=default_chunk_size,
            chunk_overlap=default_overlap
        )
        
        # Larger chunks for structured data
        self.structured = SemanticChunker(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", " | ", ", ", " "]
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Adaptively chunk based on content type."""
        if not text or not text.strip():
            return []
        
        # Check if structured data (CSV-like)
        if metadata and metadata.get('doc_type') in ['employee_record', 'financial_data']:
            # Use larger chunks for tabular data
            return self.structured.chunk(text, metadata)
        
        # Check if contains many tables/pipes
        if text.count('|') > 10 or text.count('\t') > 10:
            return self.structured.chunk(text, metadata)
        
        # Default semantic chunking
        return self.semantic.chunk(text, metadata)


def get_chunker(
    strategy: str = "semantic",
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> BaseChunker:
    """
    Factory function to get a chunker instance.
    
    Args:
        strategy: Chunking strategy ('semantic', 'llama', 'adaptive')
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        
    Returns:
        BaseChunker instance
    """
    if strategy == "llama":
        return LlamaIndexSemanticChunker(chunk_size, chunk_overlap)
    elif strategy == "adaptive":
        return AdaptiveChunker(chunk_size, chunk_overlap)
    else:  # semantic (default)
        return SemanticChunker(chunk_size, chunk_overlap)
