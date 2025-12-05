"""RAG Agent module for document retrieval."""

from .graph import rag_agent, graph
from .tools import (
    retrieve_financial_documents,
    retrieve_legal_documents,
    retrieve_hr_documents,
    retrieve_employee_records,
    retrieve_contracts,
    retrieve_litigation_records,
    search_all_documents,
    get_company_overview,
    get_vectorstore,
    COLLECTIONS,
    COMPANIES,
    normalize_company_id,
)

__all__ = [
    "rag_agent",
    "graph",
    "retrieve_financial_documents",
    "retrieve_legal_documents",
    "retrieve_hr_documents",
    "retrieve_employee_records",
    "retrieve_contracts",
    "retrieve_litigation_records",
    "search_all_documents",
    "get_company_overview",
    "get_vectorstore",
    "COLLECTIONS",
    "COMPANIES",
    "normalize_company_id",
]
