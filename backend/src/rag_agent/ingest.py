"""
RAG Ingestion Pipeline.

Orchestrates document loading, chunking, deduplication, embedding, and storage.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from src.rag_agent.base import (
    DocumentChunk, IngestionStats, DocumentCategory, DocumentType
)
from src.rag_agent.loaders import get_loader_for_file
from src.rag_agent.chunker import get_chunker
from src.rag_agent.deduplicator import HybridDeduplicator
from src.rag_agent.metadata_normalizer import normalize_metadata
from src.config.llm_config import get_embedding_model
from src.config.settings import get_settings
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Collection names
COLLECTIONS = {
    DocumentCategory.FINANCIAL: "dd_financial_docs",
    DocumentCategory.LEGAL: "dd_legal_docs",
    DocumentCategory.HR: "dd_hr_docs",
    DocumentCategory.MARKET: "dd_market_docs",
    "all": "dd_all_docs",
}


class RAGIngestionPipeline:
    """Pipeline for ingesting documents into ChromaDB."""
    
    def __init__(
        self,
        chunk_strategy: str = "adaptive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_dedup: bool = True,
        fuzzy_threshold: float = 0.8,
        batch_size: int = 128,
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            chunk_strategy: Chunking strategy ('semantic', 'llama', 'adaptive')
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            use_dedup: Enable deduplication
            fuzzy_threshold: Fuzzy dedup threshold
            batch_size: Batch size for embeddings
        """
        self.chunker = get_chunker(chunk_strategy, chunk_size, chunk_overlap)
        self.deduplicator = HybridDeduplicator(fuzzy_threshold=fuzzy_threshold) if use_dedup else None
        self.batch_size = batch_size
        self.settings = get_settings()
        
        logger.info(f"Initialized RAG ingestion pipeline with {chunk_strategy} chunking")
    
    def _get_vectorstore(self, category: DocumentCategory) -> Chroma:
        """Get or create a ChromaDB vector store for a category."""
        collection_name = COLLECTIONS.get(category, COLLECTIONS["all"])
        embeddings = get_embedding_model()
        
        persist_dir = Path(self.settings.chroma_persist_directory) / collection_name
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
    
    def ingest_file(self, file_path: Path, stats: IngestionStats) -> List[DocumentChunk]:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to file
            stats: Stats object to update
            
        Returns:
            List of processed chunks
        """
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Get appropriate loader
            loader = get_loader_for_file(file_path)
            if not loader:
                logger.warning(f"No loader found for {file_path.suffix}")
                stats.errors.append(f"No loader for {file_path.name}")
                return []
            
            # Load document chunks
            doc_chunks = loader.load(file_path)
            if not doc_chunks:
                logger.warning(f"No content extracted from {file_path.name}")
                return []
            
            processed_chunks = []
            
            for doc_chunk in doc_chunks:
                # Chunk the content
                text_chunks = self.chunker.chunk(
                    doc_chunk.content,
                    metadata=doc_chunk.metadata.to_dict()
                )
                
                # Create chunks with metadata
                for i, text in enumerate(text_chunks):
                    # Update chunk metadata
                    doc_chunk.metadata.chunk_index = i
                    doc_chunk.metadata.total_chunks = len(text_chunks)
                    
                    # Deduplication
                    if self.deduplicator:
                        if self.deduplicator.is_duplicate(text):
                            stats.chunks_deduplicated += 1
                            continue
                        chunk_hash = self.deduplicator.add(text)
                    else:
                        import hashlib
                        chunk_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                    
                    doc_chunk.metadata.chunk_hash = chunk_hash
                    
                    # Create final chunk
                    chunk = DocumentChunk(
                        content=text,
                        metadata=doc_chunk.metadata,
                    )
                    
                    processed_chunks.append(chunk)
                    stats.chunks_created += 1
                    stats.total_characters += len(text)
            
            stats.files_processed += 1
            
            # Update category stats
            if processed_chunks:
                category = processed_chunks[0].metadata.category.value
                stats.categories[category] = stats.categories.get(category, 0) + len(processed_chunks)
            
            return processed_chunks
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            stats.files_failed += 1
            stats.errors.append(f"{file_path.name}: {str(e)}")
            return []
    
    def _chunks_to_langchain_docs(self, chunks: List[DocumentChunk]) -> List[Document]:
        """Convert DocumentChunk objects to LangChain Document objects."""
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata=chunk.metadata.to_dict()
            )
            docs.append(doc)
        return docs
    
    def _store_chunks(self, chunks: List[DocumentChunk], category: DocumentCategory):
        """Store chunks in ChromaDB."""
        if not chunks:
            return
        
        try:
            vectorstore = self._get_vectorstore(category)
            
            # Convert to LangChain documents
            docs = self._chunks_to_langchain_docs(chunks)
            
            # Add in batches
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i:i + self.batch_size]
                vectorstore.add_documents(batch)
                logger.info(f"Stored batch {i//self.batch_size + 1} ({len(batch)} chunks) for {category.value}")
            
            # Also store in 'all' collection
            if category != DocumentCategory.UNKNOWN:
                all_vectorstore = self._get_vectorstore(DocumentCategory.UNKNOWN)
                for i in range(0, len(docs), self.batch_size):
                    batch = docs[i:i + self.batch_size]
                    all_vectorstore.add_documents(batch)
        
        except Exception as e:
            logger.error(f"Error storing chunks for {category}: {e}", exc_info=True)
    
    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_pattern: str = "*.*"
    ) -> IngestionStats:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory: Directory to scan
            recursive: Scan subdirectories
            file_pattern: File pattern to match
            
        Returns:
            IngestionStats with results
        """
        stats = IngestionStats()
        
        logger.info(f"Starting ingestion from {directory}")
        
        # Collect files
        if recursive:
            files = list(directory.rglob(file_pattern))
        else:
            files = list(directory.glob(file_pattern))
        
        # Filter to supported files
        supported_extensions = {'.csv', '.txt', '.md', '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        files = [f for f in files if f.suffix.lower() in supported_extensions and f.is_file()]
        
        logger.info(f"Found {len(files)} supported files")
        
        # Group chunks by category
        chunks_by_category = defaultdict(list)
        
        # Process each file
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")
            
            chunks = self.ingest_file(file_path, stats)
            
            # Group by category
            for chunk in chunks:
                chunks_by_category[chunk.metadata.category].append(chunk)
        
        # Store chunks by category
        for category, chunks in chunks_by_category.items():
            logger.info(f"Storing {len(chunks)} chunks for category {category.value}")
            self._store_chunks(chunks, category)
        
        stats.end_time = datetime.now()
        
        # Log summary
        logger.info(f"""
Ingestion Complete:
  Files processed: {stats.files_processed}
  Files failed: {stats.files_failed}
  Chunks created: {stats.chunks_created}
  Chunks deduplicated: {stats.chunks_deduplicated}
  Dedup ratio: {stats.dedup_ratio:.2%}
  Duration: {stats.duration_seconds:.2f}s
  Categories: {dict(stats.categories)}
        """)
        
        return stats
    
    def ingest_data_directory(self) -> IngestionStats:
        """Ingest all data from the standard data directory."""
        data_dir = Path(__file__).parent.parent.parent / "data" / "row_data"
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return IngestionStats()
        
        return self.ingest_directory(data_dir, recursive=True)


def ingest_all_data(
    chunk_strategy: str = "adaptive",
    chunk_size: int = 512,
    use_dedup: bool = True,
) -> IngestionStats:
    """
    Convenience function to ingest all data.
    
    Args:
        chunk_strategy: Chunking strategy
        chunk_size: Chunk size
        use_dedup: Enable deduplication
        
    Returns:
        IngestionStats
    """
    pipeline = RAGIngestionPipeline(
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        use_dedup=use_dedup,
    )
    
    return pipeline.ingest_data_directory()


if __name__ == "__main__":
    # Run ingestion
    print("Starting RAG data ingestion...")
    stats = ingest_all_data()
    
    print(f"\n{'='*60}")
    print("INGESTION SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {stats.files_processed}")
    print(f"Files failed: {stats.files_failed}")
    print(f"Chunks created: {stats.chunks_created}")
    print(f"Chunks deduplicated: {stats.chunks_deduplicated}")
    print(f"Dedup ratio: {stats.dedup_ratio:.2%}")
    print(f"Total characters: {stats.total_characters:,}")
    print(f"Duration: {stats.duration_seconds:.2f}s")
    print(f"\nBy Category:")
    for cat, count in sorted(stats.categories.items()):
        print(f"  {cat}: {count}")
    
    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:10]:
            print(f"  - {error}")
