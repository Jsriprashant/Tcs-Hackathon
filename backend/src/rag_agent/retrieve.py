"""
RAG Retrieval Module.

Implements hybrid search (semantic + BM25), MMR reranking, and filtering.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from functools import lru_cache
import time
import numpy as np

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.rag_agent.base import DocumentCategory, DocumentType, RetrievalResult
from src.rag_agent.metadata_normalizer import normalize_doc_type, normalize_category, normalize_metadata
from src.config.llm_config import get_embedding_model, get_llm
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


class HybridRetriever:
    """Hybrid retriever combining semantic and BM25 search with true hybrid retrieval."""
    
    def __init__(
        self,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_mmr: bool = True,
        mmr_diversity: float = 0.3,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_weight: Weight for semantic search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            use_mmr: Use MMR for reranking
            mmr_diversity: Diversity parameter for MMR (0=relevance, 1=diversity)
        """
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.use_mmr = use_mmr
        self.mmr_diversity = mmr_diversity
        self.settings = get_settings()
        
        # Cache for BM25 indices per collection
        self._bm25_cache: Dict[str, Tuple[BM25Okapi, List[Document]]] = {}
        # Cache for collection documents
        self._collection_docs_cache: Dict[str, List[Document]] = {}
    
    def _get_vectorstore(self, category: Optional[DocumentCategory] = None, strict: bool = False) -> Chroma:
        """
        Get ChromaDB vector store.
        
        Args:
            category: Document category for collection selection
            strict: If True, raise error if collection not found instead of fallback
            
        Returns:
            Chroma vectorstore instance
        """
        if category and category != DocumentCategory.UNKNOWN:
            collection_name = COLLECTIONS.get(category, COLLECTIONS["all"])
        else:
            collection_name = COLLECTIONS["all"]
        
        embeddings = get_embedding_model()
        persist_dir = Path(self.settings.chroma_persist_directory) / collection_name
        
        if not persist_dir.exists():
            if strict:
                raise ValueError(f"Collection '{collection_name}' not found at {persist_dir}. "
                               f"Please ensure the collection exists or use category=None for 'all'.")
            logger.warning(f"Collection {collection_name} not found at {persist_dir}, using 'all'")
            collection_name = COLLECTIONS["all"]
            persist_dir = Path(self.settings.chroma_persist_directory) / collection_name
        
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
    
    def _get_collection_documents(self, category: Optional[DocumentCategory], filter_dict: Dict[str, Any]) -> List[Document]:
        """
        Get all documents from a collection for BM25 indexing.
        
        Args:
            category: Document category
            filter_dict: Metadata filters to apply
            
        Returns:
            List of documents from the collection
        """
        cache_key = f"{category}_{hash(frozenset(filter_dict.items()) if filter_dict else '')}"
        
        if cache_key in self._collection_docs_cache:
            return self._collection_docs_cache[cache_key]
        
        try:
            vectorstore = self._get_vectorstore(category)
            # Fetch a larger set for BM25 - get all or a large sample
            # Use a dummy query to fetch documents (we'll filter by metadata)
            if filter_dict:
                docs = vectorstore.similarity_search("", k=1000, filter=filter_dict)
            else:
                docs = vectorstore.similarity_search("", k=1000)
            
            self._collection_docs_cache[cache_key] = docs
            logger.info(f"Cached {len(docs)} documents for BM25 index (category={category})")
            return docs
        except Exception as e:
            logger.warning(f"Could not fetch collection documents: {e}")
            return []
    
    def _build_bm25_index(self, docs: List[Document]) -> BM25Okapi:
        """Build BM25 index from documents."""
        if not docs:
            return None
        # Tokenize documents (simple whitespace tokenization with basic cleaning)
        tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
        return BM25Okapi(tokenized_corpus)
    
    def _get_bm25_scores(
        self,
        query: str,
        docs: List[Document],
        collection_key: str
    ) -> List[float]:
        """Get BM25 scores for query against documents."""
        # Check cache
        if collection_key not in self._bm25_cache:
            bm25 = self._build_bm25_index(docs)
            self._bm25_cache[collection_key] = (bm25, docs)
        else:
            bm25, cached_docs = self._bm25_cache[collection_key]
            # Rebuild if docs changed
            if len(cached_docs) != len(docs):
                bm25 = self._build_bm25_index(docs)
                self._bm25_cache[collection_key] = (bm25, docs)
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        return scores.tolist()
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _apply_mmr(
        self,
        docs: List[Document],
        scores: List[float],
        query: str,
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """Apply Maximal Marginal Relevance for diversity."""
        if len(docs) <= top_k:
            return list(zip(docs, scores))
        
        # Simple MMR implementation
        selected = []
        remaining = list(zip(docs, scores))
        remaining.sort(key=lambda x: x[1], reverse=True)
        
        # Select first (most relevant)
        selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for idx, (doc, relevance) in enumerate(remaining):
                # Calculate diversity penalty
                max_similarity = 0.0
                for selected_doc, _ in selected:
                    # Simple text overlap similarity
                    overlap = len(set(doc.page_content.lower().split()) & 
                                set(selected_doc.page_content.lower().split()))
                    total = len(set(doc.page_content.lower().split()) | 
                              set(selected_doc.page_content.lower().split()))
                    similarity = overlap / total if total > 0 else 0
                    max_similarity = max(max_similarity, similarity)
                  # MMR score: balance relevance and diversity
                mmr_score = (self.mmr_diversity * relevance - 
                           (1 - self.mmr_diversity) * max_similarity)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def search(
        self,
        query: str,
        category: Optional[DocumentCategory] = None,
        company_id: Optional[str] = None,
        doc_type: Optional[DocumentType] = None,
        top_k: int = 10,
        fetch_k: int = 30,
        strict_category: bool = False,
    ) -> List[RetrievalResult]:
        """
        True hybrid search with independent semantic and BM25 retrieval.
        
        Args:
            query: Search query
            category: Optional category filter
            company_id: Optional company filter
            doc_type: Optional document type filter
            top_k: Number of results to return
            fetch_k: Number of candidates to fetch for reranking
            strict_category: If True, raise error if category collection doesn't exist
            
        Returns:
            List of RetrievalResult objects
        """
        start_time = time.time()
        
        try:
            vectorstore = self._get_vectorstore(category, strict=strict_category)
            
            # Build metadata filter with normalized values
            filter_dict = {}
            if company_id:
                filter_dict["company_id"] = company_id
            if doc_type:
                # Normalize doc_type to canonical value
                normalized_doc_type = normalize_doc_type(doc_type)
                filter_dict["doc_type"] = normalized_doc_type.value
            
            # === SEMANTIC SEARCH ===
            try:
                if filter_dict:
                    semantic_docs_with_scores = vectorstore.similarity_search_with_score(
                        query,
                        k=fetch_k,
                        filter=filter_dict
                    )
                else:
                    semantic_docs_with_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}, trying without filter")
                semantic_docs_with_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
            
            # Convert similarity to relevance (lower distance = higher relevance)
            semantic_results = {}
            for doc, score in semantic_docs_with_scores:
                doc_id = hash(doc.page_content)
                relevance = 1.0 / (1.0 + score)
                semantic_results[doc_id] = (doc, relevance)
            
            # === BM25 SEARCH (True Hybrid - independent from semantic) ===
            bm25_results = {}
            collection_key = f"{category}_{company_id}_{doc_type}"
            
            # Get collection documents for BM25
            collection_docs = self._get_collection_documents(category, filter_dict)
            
            if collection_docs:
                # Build or get cached BM25 index
                if collection_key not in self._bm25_cache:
                    bm25 = self._build_bm25_index(collection_docs)
                    self._bm25_cache[collection_key] = (bm25, collection_docs)
                else:
                    bm25, cached_docs = self._bm25_cache[collection_key]
                    if len(cached_docs) != len(collection_docs):
                        bm25 = self._build_bm25_index(collection_docs)
                        self._bm25_cache[collection_key] = (bm25, collection_docs)
                
                if bm25:
                    tokenized_query = query.lower().split()
                    bm25_scores = bm25.get_scores(tokenized_query)
                    
                    # Get top-k BM25 results
                    top_bm25_indices = np.argsort(bm25_scores)[-fetch_k:][::-1]
                    for idx in top_bm25_indices:
                        if bm25_scores[idx] > 0:
                            doc = collection_docs[idx]
                            doc_id = hash(doc.page_content)
                            bm25_results[doc_id] = (doc, bm25_scores[idx])
            
            # === MERGE RESULTS ===
            all_doc_ids = set(semantic_results.keys()) | set(bm25_results.keys())
            
            if not all_doc_ids:
                logger.warning(f"No documents found for query: {query[:50]}...")
                return []
            
            # Normalize scores
            semantic_scores_raw = [r[1] for r in semantic_results.values()] if semantic_results else [0]
            bm25_scores_raw = [r[1] for r in bm25_results.values()] if bm25_results else [0]
            
            semantic_min, semantic_max = min(semantic_scores_raw), max(semantic_scores_raw)
            bm25_min, bm25_max = min(bm25_scores_raw), max(bm25_scores_raw)
            
            def normalize(val, min_v, max_v):
                if max_v == min_v:
                    return 1.0
                return (val - min_v) / (max_v - min_v)
            
            # Compute hybrid scores
            merged_results = []
            for doc_id in all_doc_ids:
                # Get doc from either result set
                if doc_id in semantic_results:
                    doc, sem_score = semantic_results[doc_id]
                    sem_norm = normalize(sem_score, semantic_min, semantic_max)
                else:
                    sem_norm = 0.0
                    doc = bm25_results[doc_id][0]
                
                if doc_id in bm25_results:
                    _, bm25_score = bm25_results[doc_id]
                    bm25_norm = normalize(bm25_score, bm25_min, bm25_max)
                else:
                    bm25_norm = 0.0
                
                hybrid_score = self.semantic_weight * sem_norm + self.bm25_weight * bm25_norm
                merged_results.append((doc, hybrid_score))
            
            # Sort by hybrid score
            merged_results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply MMR if enabled
            if self.use_mmr and len(merged_results) > top_k:
                results = self._apply_mmr(
                    [r[0] for r in merged_results], 
                    [r[1] for r in merged_results], 
                    query, 
                    top_k
                )
            else:
                results = merged_results[:top_k]
            
            # Convert to RetrievalResult with normalized metadata
            retrieval_results = []
            for doc, score in results:
                # Normalize document metadata
                normalized_meta = normalize_metadata(doc.metadata) if doc.metadata else {}
                result = RetrievalResult(
                    content=doc.page_content,
                    score=score,
                    metadata=normalized_meta,
                    retrieval_method="hybrid"
                )
                retrieval_results.append(result)
            
            elapsed = time.time() - start_time
            logger.info(f"Retrieved {len(retrieval_results)} results in {elapsed:.3f}s "
                       f"(semantic: {len(semantic_results)}, bm25: {len(bm25_results)})")
            
            return retrieval_results
        
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}", exc_info=True)
            return []
    
    def clear_cache(self):
        """Clear BM25 and document caches."""
        self._bm25_cache.clear()
        self._collection_docs_cache.clear()


class RAGRetriever:
    """Main RAG retriever with generation capabilities."""
    
    def __init__(self):
        """Initialize RAG retriever."""
        self.hybrid_retriever = HybridRetriever()
        self.llm = get_llm(temperature=0.1)
    
    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        company_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query
            category: Category filter ('financial', 'legal', 'hr', 'market')
            company_id: Company filter
            doc_type: Document type filter
            top_k: Number of results
            
        Returns:
            List of RetrievalResult
        """
        # Use metadata normalizer for robust enum conversion
        category_enum = normalize_category(category) if category else None
        if category_enum == DocumentCategory.UNKNOWN and category:
            logger.warning(f"Category '{category}' not recognized, using 'all' collection")
            category_enum = None
        
        doc_type_enum = normalize_doc_type(doc_type) if doc_type else None
        if doc_type_enum == DocumentType.UNKNOWN and doc_type:
            logger.info(f"Doc type '{doc_type}' normalized to UNKNOWN")
        
        return self.hybrid_retriever.search(
            query=query,
            category=category_enum,
            company_id=company_id,
            doc_type=doc_type_enum,
            top_k=top_k,
        )
    
    def generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate answer using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer
        """
        if not contexts:
            return "I couldn't find relevant information to answer your question."
        
        # Build context string
        context_str = "\n\n".join([
            f"[Document {i+1} - Score: {ctx.score:.3f}]\n{ctx.content}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Default system prompt
        if not system_prompt:
            system_prompt = """You are a helpful assistant for M&A due diligence analysis.
Answer the question based on the provided context documents.
Be specific and cite information from the documents when possible.
If the context doesn't contain enough information, say so."""
        
        # Build prompt
        prompt = f"""{system_prompt}

Context Documents:
{context_str}

Question: {query}

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def query(
        self,
        query: str,
        category: Optional[str] = None,
        company_id: Optional[str] = None,
        top_k: int = 5,
        include_generation: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RAG query: retrieve + generate.
        
        Args:
            query: User query
            category: Category filter
            company_id: Company filter
            top_k: Number of contexts
            include_generation: Generate answer
            
        Returns:
            Dict with 'contexts', 'answer', 'latency'
        """
        start_time = time.time()
        
        # Retrieve
        contexts = self.retrieve(
            query=query,
            category=category,
            company_id=company_id,
            top_k=top_k
        )
        
        result = {
            "contexts": [ctx.to_dict() for ctx in contexts],
            "query": query,
        }
        
        # Generate
        if include_generation and contexts:
            answer = self.generate(query, contexts)
            result["answer"] = answer
        
        result["latency"] = time.time() - start_time
        
        return result


# Singleton instance
_retriever_instance = None


def get_retriever() -> RAGRetriever:
    """Get singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance
