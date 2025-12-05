"""
Deduplication utilities for RAG ingestion.

Implements exact hash and fuzzy MinHash-based deduplication.
"""

import hashlib
from typing import Set, Optional
from datasketch import MinHash, MinHashLSH

from src.rag_agent.base import BaseDeduplicator
from src.common.logging_config import get_logger

logger = get_logger(__name__)


class ExactDeduplicator(BaseDeduplicator):
    """Exact hash-based deduplication using SHA256."""
    
    def __init__(self):
        """Initialize the exact deduplicator."""
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of normalized text."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is an exact duplicate."""
        text_hash = self._compute_hash(text)
        return text_hash in self.seen_hashes
    
    def add(self, text: str) -> str:
        """Add text to dedup index and return hash."""
        text_hash = self._compute_hash(text)
        if text_hash in self.seen_hashes:
            self.duplicate_count += 1
        else:
            self.seen_hashes.add(text_hash)
        return text_hash
    
    def clear(self):
        """Clear the deduplication index."""
        self.seen_hashes.clear()
        self.duplicate_count = 0
    
    @property
    def total_unique(self) -> int:
        """Get total unique items."""
        return len(self.seen_hashes)


class FuzzyDeduplicator(BaseDeduplicator):
    """Fuzzy deduplication using MinHash LSH."""
    
    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram_size: int = 3
    ):
        """
        Initialize fuzzy deduplicator.
        
        Args:
            threshold: Jaccard similarity threshold (0.0-1.0)
            num_perm: Number of permutations for MinHash
            ngram_size: Size of character n-grams
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}
        self.duplicate_count = 0
        self.id_counter = 0
    
    def _create_minhash(self, text: str) -> MinHash:
        """Create MinHash from text using character n-grams."""
        minhash = MinHash(num_perm=self.num_perm)
        
        # Normalize text
        normalized = text.strip().lower()
        
        # Create character n-grams
        for i in range(len(normalized) - self.ngram_size + 1):
            ngram = normalized[i:i + self.ngram_size]
            minhash.update(ngram.encode('utf-8'))
        
        return minhash
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a fuzzy duplicate."""
        if not text.strip():
            return False
        
        minhash = self._create_minhash(text)
        # Query LSH for similar items
        results = self.lsh.query(minhash)
        
        return len(results) > 0
    
    def add(self, text: str) -> str:
        """Add text to LSH index and return unique ID."""
        if not text.strip():
            return ""
        
        minhash = self._create_minhash(text)
        
        # Check if similar document exists
        if self.is_duplicate(text):
            self.duplicate_count += 1
            # Return existing similar doc ID
            results = self.lsh.query(minhash)
            return results[0] if results else ""
        
        # Add new document
        doc_id = f"doc_{self.id_counter}"
        self.id_counter += 1
        self.lsh.insert(doc_id, minhash)
        self.minhashes[doc_id] = minhash
        
        return doc_id
    
    def clear(self):
        """Clear the LSH index."""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes.clear()
        self.duplicate_count = 0
        self.id_counter = 0
    
    @property
    def total_unique(self) -> int:
        """Get total unique items."""
        return len(self.minhashes)


class HybridDeduplicator(BaseDeduplicator):
    """Hybrid deduplicator using both exact and fuzzy methods."""
    
    def __init__(
        self,
        fuzzy_threshold: float = 0.8,
        num_perm: int = 128,
        ngram_size: int = 3
    ):
        """Initialize hybrid deduplicator."""
        self.exact = ExactDeduplicator()
        self.fuzzy = FuzzyDeduplicator(
            threshold=fuzzy_threshold,
            num_perm=num_perm,
            ngram_size=ngram_size
        )
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate using exact first, then fuzzy."""
        # Fast exact check first
        if self.exact.is_duplicate(text):
            return True
        
        # Slower fuzzy check
        return self.fuzzy.is_duplicate(text)
    
    def add(self, text: str) -> str:
        """Add text to both indexes."""
        exact_hash = self.exact.add(text)
        
        # Only add to fuzzy if not exact duplicate
        if exact_hash not in self.exact.seen_hashes or len(self.exact.seen_hashes) == 1:
            fuzzy_id = self.fuzzy.add(text)
            return exact_hash  # Return exact hash as primary ID
        
        return exact_hash
    
    def clear(self):
        """Clear both indexes."""
        self.exact.clear()
        self.fuzzy.clear()
    
    @property
    def total_duplicates(self) -> int:
        """Get total duplicates found."""
        return self.exact.duplicate_count + self.fuzzy.duplicate_count
    
    @property
    def exact_duplicates(self) -> int:
        """Get exact duplicates count."""
        return self.exact.duplicate_count
    
    @property
    def fuzzy_duplicates(self) -> int:
        """Get fuzzy duplicates count."""
        return self.fuzzy.duplicate_count
