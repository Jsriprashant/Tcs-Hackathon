"""RAG retrieval utilities for Legal Agent MVP.

This module wraps the RAG agent tools for legal document retrieval.
No fallbacks - uses direct RAG tool invocation.
"""

from typing import Tuple

from src.rag_agent.tools import (
    retrieve_legal_documents,
    retrieve_contracts,
    retrieve_litigation_records,
    normalize_company_id,
)
from src.common.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Category-specific retrieval configuration
CATEGORY_CONFIG = {
    "litigation": {
        "keywords": "litigation lawsuit pending court case judgment penalty regulatory SEBI SEC EEOC GDPR fine enforcement investigation",
        "k": 10,
    },
    "contracts": {
        "keywords": "contract agreement debt covenant change control termination vendor customer assignment license indemnification",
        "k": 15,
    },
    "ip": {
        "keywords": "patent trademark IP intellectual property license open source GPL copyright trade secret assignment infringement",
        "k": 10,
    },
}

# Benchmark queries for comparison
BENCHMARK_QUERIES = {
    "litigation": "regulatory penalty matrix enforcement order judgment standard compliance violation fine",
    "contracts": "contract template standard terms SEC EDGAR agreement best practice change of control",
    "ip": "WIPO patent trademark open source license GPL AGPL compliance standard assignment",
}


# =============================================================================
# RETRIEVAL FUNCTIONS
# =============================================================================

def retrieve_company_docs(company_id: str, category: str) -> str:
    """
    Retrieve company-specific documents for a category.
    
    Args:
        company_id: Company identifier (BBD, SUPERNOVA, etc.)
        category: One of "litigation", "contracts", "ip"
    
    Returns:
        Markdown-formatted document content from ChromaDB
    """
    normalized_id = normalize_company_id(company_id)
    config = CATEGORY_CONFIG[category]
    
    logger.info(f"Retrieving {category} docs for {normalized_id} (k={config['k']})")
    
    if category == "litigation":
        result = retrieve_litigation_records.invoke({
            "company_id": normalized_id,
            "k": config["k"],
        })
    
    elif category == "contracts":
        result = retrieve_contracts.invoke({
            "company_id": normalized_id,
            "contract_type": "",  # All contract types
            "k": config["k"],
        })
    
    else:  # ip
        result = retrieve_legal_documents.invoke({
            "company_id": normalized_id,
            "query": config["keywords"],
            "doc_type": "ip",
            "k": config["k"],
        })
    
    logger.info(f"Retrieved {len(result)} chars for {category}")
    return result


def retrieve_benchmark_docs(category: str) -> str:
    """
    Retrieve benchmark/standard documents for comparison.
    
    Benchmarks are stored without company_id filter.
    These represent industry standards to compare company docs against.
    
    Args:
        category: One of "litigation", "contracts", "ip"
    
    Returns:
        Markdown-formatted benchmark content
    """
    query = BENCHMARK_QUERIES[category]
    
    logger.info(f"Retrieving {category} benchmark docs")
    
    result = retrieve_legal_documents.invoke({
        "company_id": "",  # No company filter for benchmarks
        "query": query,
        "doc_type": "",
        "k": 5,
    })
    
    logger.info(f"Retrieved {len(result)} chars for {category} benchmarks")
    return result


def retrieve_for_category(company_id: str, category: str) -> Tuple[str, str]:
    """
    Retrieve both company docs and benchmark docs for a category.
    
    This is the main function called by analysis nodes.
    
    Args:
        company_id: Company identifier
        category: Category to analyze (litigation, contracts, ip)
    
    Returns:
        Tuple of (company_docs, benchmark_docs) as Markdown strings
    
    Example:
        >>> company_docs, benchmark_docs = retrieve_for_category("BBD", "litigation")
        >>> print(len(company_docs), len(benchmark_docs))
    """
    company_docs = retrieve_company_docs(company_id, category)
    benchmark_docs = retrieve_benchmark_docs(category)
    
    return company_docs, benchmark_docs


def get_normalized_company_id(company_id: str) -> str:
    """
    Normalize company ID to standard format.
    
    Wrapper around RAG agent's normalize function.
    
    Args:
        company_id: Raw company identifier
    
    Returns:
        Normalized company ID (SUPERNOVA, BBD, etc.)
    """
    return normalize_company_id(company_id)
