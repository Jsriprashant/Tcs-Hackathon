# filepath: c:\Users\GenAIBLRANCUSR25.01HW2562306\Desktop\application_v1\Tcs-Hackathon\backend\src\rag_agent\tools.py
"""RAG Agent tools for document retrieval from ChromaDB vector stores."""

import os
from typing import Any, Optional, List
from pathlib import Path

from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma

from src.config.llm_config import get_embedding_model
from src.config.settings import get_settings
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Collection names for ChromaDB
COLLECTIONS = {
    "financial": "dd_financial_docs",
    "legal": "dd_legal_docs",
    "hr": "dd_hr_docs",
    "all": "dd_all_docs",
}

# Company mappings
COMPANIES = {
    "BBD": {"name": "BBD Ltd", "aliases": ["BBD_LTD", "BBD Software", "BBD_Software"]},
    "XYZ": {"name": "XYZ Ltd", "aliases": ["XYZ_LTD", "XYZ LTD"]},
    "SUPERNOVA": {"name": "Supernova Inc", "aliases": ["Supernova", "Supernova_"]},
    "RASPUTIN": {"name": "Rasputin Petroleum Ltd", "aliases": ["Rasputin", "Rasputil_Petroleum", "Rasputin_"]},
    "TECHNOBOX": {"name": "Techno Box Inc", "aliases": ["Techno_Box", "TechnoBox"]},
}


def get_chroma_client():
    """Get ChromaDB persist directory."""
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


def get_vectorstore(collection_name: str) -> Chroma:
    """Get or create a ChromaDB vector store."""
    embeddings = get_embedding_model()
    persist_directory = str(get_chroma_client() / collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def normalize_company_id(company_id: str) -> str:
    """Normalize company ID to standard format."""
    company_id = company_id.upper().strip()
    for standard_id, info in COMPANIES.items():
        if company_id == standard_id:
            return standard_id
        for alias in info["aliases"]:
            if alias.upper() in company_id or company_id in alias.upper():
                return standard_id
    return company_id


@tool
def retrieve_financial_documents(
    company_id: str,
    query: str = "",
    k: int = 5,
) -> str:
    """
    Retrieve financial documents for a company from the vector store.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
        query: Optional search query to filter results
        k: Number of documents to retrieve
    
    Returns:
        Retrieved financial documents content
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["financial"])
        normalized_id = normalize_company_id(company_id)
        
        search_query = f"{normalized_id} {query}".strip() if query else f"{normalized_id} financial data"
        
        # Try with company filter first
        docs = vectorstore.similarity_search(
            search_query,
            k=k,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            # Fallback without filter
            docs = vectorstore.similarity_search(search_query, k=k)
        
        if not docs:
            return f"No financial documents found for {company_id}"
        
        result = f"## Financial Documents for {normalized_id}\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"### Document {i}\n"
            result += f"**Source:** {doc.metadata.get('filename', 'Unknown')}\n"
            result += f"**Type:** {doc.metadata.get('doc_type', 'Unknown')}\n\n"
            result += f"{doc.page_content}\n\n---\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving financial documents: {e}")
        return f"Error retrieving financial documents: {str(e)}"


@tool
def retrieve_legal_documents(
    company_id: str,
    query: str = "",
    doc_type: str = "",
    k: int = 5,
) -> str:
    """
    Retrieve legal documents for a company from the vector store.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
        query: Optional search query to filter results
        doc_type: Type of legal document (contract, litigation, ip, compliance)
        k: Number of documents to retrieve
    
    Returns:
        Retrieved legal documents content
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        search_query = f"{normalized_id} {doc_type} {query}".strip()
        if not search_query:
            search_query = f"{normalized_id} legal document"
        
        # Build filter using ChromaDB $and syntax for multiple conditions
        if doc_type and normalized_id:
            filter_dict = {
                "$and": [
                    {"company_id": normalized_id},
                    {"doc_type": doc_type}
                ]
            }
        elif normalized_id:
            filter_dict = {"company_id": normalized_id}
        else:
            filter_dict = None
        
        docs = vectorstore.similarity_search(
            search_query,
            k=k,
            filter=filter_dict
        )
        
        if not docs:
            docs = vectorstore.similarity_search(search_query, k=k)
        
        if not docs:
            return f"No legal documents found for {company_id}"
        
        result = f"## Legal Documents for {normalized_id}\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"### Document {i}\n"
            result += f"**Source:** {doc.metadata.get('filename', 'Unknown')}\n"
            result += f"**Type:** {doc.metadata.get('doc_type', 'Unknown')}\n"
            result += f"**Category:** {doc.metadata.get('category', 'legal')}\n\n"
            result += f"{doc.page_content}\n\n---\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving legal documents: {e}")
        return f"Error retrieving legal documents: {str(e)}"


@tool
def retrieve_hr_documents(
    company_id: str,
    query: str = "",
    doc_type: str = "",
    k: int = 5,
) -> str:
    """
    Retrieve HR documents for a company from the vector store.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
        query: Optional search query to filter results
        doc_type: Type of HR document (employee_data, policy, handbook)
        k: Number of documents to retrieve
    
    Returns:
        Retrieved HR documents content
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        search_query = f"{normalized_id} {doc_type} {query}".strip()
        if not search_query:
            search_query = f"{normalized_id} HR employee data"
        
        docs = vectorstore.similarity_search(
            search_query,
            k=k,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(search_query, k=k)
        
        if not docs:
            return f"No HR documents found for {company_id}"
        
        result = f"## HR Documents for {normalized_id}\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"### Document {i}\n"
            result += f"**Source:** {doc.metadata.get('filename', 'Unknown')}\n"
            result += f"**Type:** {doc.metadata.get('doc_type', 'Unknown')}\n\n"
            result += f"{doc.page_content}\n\n---\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving HR documents: {e}")
        return f"Error retrieving HR documents: {str(e)}"


@tool
def retrieve_employee_records(
    company_id: str,
    department: str = "",
    k: int = 10,
) -> str:
    """
    Retrieve employee records for a company.
    
    Args:
        company_id: Company identifier
        department: Optional department filter
        k: Number of records to retrieve
    
    Returns:
        Employee records with key details
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["hr"])
        normalized_id = normalize_company_id(company_id)
        
        search_query = f"{normalized_id} employee {department}".strip()
        
        # Build filter using ChromaDB $and syntax for multiple conditions
        filter_conditions = [
            {"company_id": normalized_id},
            {"doc_type": "employee_record"}
        ]
        if department:
            filter_conditions.append({"department": department})
        
        filter_dict = {"$and": filter_conditions}
        
        docs = vectorstore.similarity_search(
            search_query,
            k=k,
            filter=filter_dict
        )
        
        if not docs:
            # Fallback search
            docs = vectorstore.similarity_search(
                f"{normalized_id} employee record",
                k=k
            )
        
        if not docs:
            return f"No employee records found for {company_id}"
        
        result = f"## Employee Records for {normalized_id}\n\n"
        result += f"**Records Found:** {len(docs)}\n\n"
        
        for doc in docs:
            result += f"{doc.page_content}\n---\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving employee records: {e}")
        return f"Error retrieving employee records: {str(e)}"


@tool
def retrieve_contracts(
    company_id: str,
    contract_type: str = "",
    k: int = 5,
) -> str:
    """
    Retrieve contract documents for a company.
    
    Args:
        company_id: Company identifier
        contract_type: Type of contract (customer, vendor, employment, license)
        k: Number of contracts to retrieve
    
    Returns:
        Contract documents with key terms
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        search_query = f"{normalized_id} contract {contract_type} agreement".strip()
        
        docs = vectorstore.similarity_search(
            search_query,
            k=k,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                f"contract agreement {contract_type}",
                k=k
            )
        
        if not docs:
            return f"No contracts found for {company_id}"
        
        result = f"## Contracts for {normalized_id}\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"### Contract {i}\n"
            result += f"**Source:** {doc.metadata.get('filename', 'Unknown')}\n"
            result += f"**Type:** {doc.metadata.get('doc_type', 'Unknown')}\n\n"
            result += f"{doc.page_content}\n\n---\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving contracts: {e}")
        return f"Error retrieving contracts: {str(e)}"


@tool
def retrieve_litigation_records(
    company_id: str,
    k: int = 5,
) -> str:
    """
    Retrieve litigation and legal dispute records for a company.
    
    Args:
        company_id: Company identifier
        k: Number of records to retrieve
    
    Returns:
        Litigation records and case details
    """
    try:
        vectorstore = get_vectorstore(COLLECTIONS["legal"])
        normalized_id = normalize_company_id(company_id)
        
        search_query = f"{normalized_id} litigation lawsuit court case dispute"
        
        docs = vectorstore.similarity_search(
            search_query,
            k=k,
            filter={"company_id": normalized_id}
        )
        
        if not docs:
            docs = vectorstore.similarity_search(
                "litigation lawsuit court judgment penalty",
                k=k
            )
        
        if not docs:
            return f"No litigation records found for {company_id}"
        
        result = f"## Litigation Records for {normalized_id}\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"### Case {i}\n"
            result += f"**Source:** {doc.metadata.get('filename', 'Unknown')}\n"
            result += f"**Type:** {doc.metadata.get('doc_type', 'Unknown')}\n\n"
            result += f"{doc.page_content}\n\n---\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving litigation records: {e}")
        return f"Error retrieving litigation records: {str(e)}"


@tool
def search_all_documents(
    query: str,
    company_id: str = "",
    k: int = 10,
) -> str:
    """
    Search across all document collections (financial, legal, HR).
    
    Args:
        query: Search query
        company_id: Optional company filter
        k: Number of results per collection
    
    Returns:
        Combined search results from all collections
    """
    try:
        results = []
        normalized_id = normalize_company_id(company_id) if company_id else ""
        
        for category, collection_name in COLLECTIONS.items():
            if category == "all":
                continue
                
            try:
                vectorstore = get_vectorstore(collection_name)
                search_query = f"{normalized_id} {query}".strip() if normalized_id else query
                
                if normalized_id:
                    docs = vectorstore.similarity_search(
                        search_query,
                        k=k // 3,
                        filter={"company_id": normalized_id}
                    )
                else:
                    docs = vectorstore.similarity_search(search_query, k=k // 3)
                
                if docs:
                    results.append(f"\n## {category.upper()} Documents\n")
                    for doc in docs:
                        results.append(f"- **{doc.metadata.get('filename', 'Unknown')}**: {doc.page_content[:300]}...")
            
            except Exception as e:
                logger.warning(f"Error searching {collection_name}: {e}")
                continue
        
        if not results:
            return f"No documents found matching: {query}"
        
        return "\n".join(results)
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return f"Error searching documents: {str(e)}"


@tool
def get_company_overview(
    company_id: str,
) -> str:
    """
    Get a comprehensive overview of all available documents for a company.
    
    Args:
        company_id: Company identifier (BBD, XYZ, SUPERNOVA, RASPUTIN, TECHNOBOX)
    
    Returns:
        Summary of available documents by category
    """
    try:
        normalized_id = normalize_company_id(company_id)
        company_info = COMPANIES.get(normalized_id, {"name": company_id, "aliases": []})
        
        overview = f"# Company Overview: {company_info['name']}\n\n"
        overview += f"**ID:** {normalized_id}\n"
        overview += f"**Aliases:** {', '.join(company_info['aliases'])}\n\n"
        
        doc_counts = {}
        
        for category, collection_name in COLLECTIONS.items():
            if category == "all":
                continue
            
            try:
                vectorstore = get_vectorstore(collection_name)
                docs = vectorstore.similarity_search(
                    f"{normalized_id}",
                    k=20,
                    filter={"company_id": normalized_id}
                )
                doc_counts[category] = len(docs)
                
                if docs:
                    overview += f"## {category.capitalize()} Documents ({len(docs)} found)\n"
                    doc_types = {}
                    for doc in docs:
                        dt = doc.metadata.get('doc_type', 'unknown')
                        doc_types[dt] = doc_types.get(dt, 0) + 1
                    
                    for dt, count in doc_types.items():
                        overview += f"- {dt}: {count}\n"
                    overview += "\n"
            
            except Exception as e:
                logger.warning(f"Error checking {collection_name}: {e}")
                overview += f"## {category.capitalize()} Documents\nError: {e}\n\n"
        
        total = sum(doc_counts.values())
        overview += f"## Summary\n**Total Documents:** {total}\n"
        
        return overview
        
    except Exception as e:
        logger.error(f"Error getting company overview: {e}")
        return f"Error getting company overview: {str(e)}"


# Export all RAG tools
rag_tools = [
    retrieve_financial_documents,
    retrieve_legal_documents,
    retrieve_hr_documents,
    retrieve_employee_records,
    retrieve_contracts,
    retrieve_litigation_records,
    search_all_documents,
    get_company_overview,
]
