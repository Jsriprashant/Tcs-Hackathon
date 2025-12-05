"""
Document Loader for M&A Due Diligence Platform.

Loads actual company documents into ChromaDB vector stores for RAG retrieval.
Supports: CSV, MD, TXT, PDF files
"""

import os
import json
import csv
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.config.llm_config import get_embedding_model
from src.config.settings import get_settings
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Base path for raw data
RAW_DATA_PATH = Path(__file__).parent / "row_data"

# Collection names for ChromaDB
COLLECTIONS = {
    "financial": "dd_financial_docs",
    "legal": "dd_legal_docs",
    "hr": "dd_hr_docs",
    "all": "dd_all_docs",
}

# Company mappings based on available data
COMPANIES = {
    "BBD": {"name": "BBD Ltd", "aliases": ["BBD_LTD", "BBD Software", "BBD_Software"]},
    "XYZ": {"name": "XYZ Ltd", "aliases": ["XYZ_LTD", "XYZ LTD"]},
    "SUPERNOVA": {"name": "Supernova Inc", "aliases": ["Supernova", "Supernova_"]},
    "RASPUTIN": {"name": "Rasputin Petroleum Ltd", "aliases": ["Rasputin", "Rasputil_Petroleum", "Rasputin_"]},
    "TECHNOBOX": {"name": "Techno Box Inc", "aliases": ["Techno_Box", "TechnoBox"]},
}


def get_chroma_client():
    """Get ChromaDB persistent client."""
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


def get_vectorstore(collection_name: str) -> Chroma:
    """Get or create a ChromaDB vector store."""
    settings = get_settings()
    embeddings = get_embedding_model()
    
    persist_directory = str(get_chroma_client() / collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def identify_company(filename: str, content: str = "") -> str:
    """Identify which company a document belongs to."""
    text_to_check = (filename + " " + content[:500]).upper()
    
    for company_id, info in COMPANIES.items():
        if company_id in text_to_check:
            return company_id
        for alias in info["aliases"]:
            if alias.upper() in text_to_check:
                return company_id
    
    return "UNKNOWN"


def read_csv_file(filepath: Path) -> list[Document]:
    """Read CSV file and convert to documents."""
    documents = []
    filename = filepath.name
    company_id = identify_company(filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip the filepath comment line if present
            first_line = f.readline()
            if first_line.startswith('//'):
                pass  # Skip comment line
            else:
                f.seek(0)  # Reset to beginning
            
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                return documents
            
            # Determine document type from filename
            doc_type = "financial_data"
            if "balance" in filename.lower():
                doc_type = "balance_sheet"
            elif "income" in filename.lower():
                doc_type = "income_statement"
            elif "cashflow" in filename.lower() or "cash_flow" in filename.lower():
                doc_type = "cash_flow_statement"
            elif "employee" in filename.lower():
                doc_type = "employee_data"
            
            # Create a summary document
            headers = list(rows[0].keys()) if rows else []
            summary_content = f"# {filename}\n\n"
            summary_content += f"**Company:** {company_id}\n"
            summary_content += f"**Document Type:** {doc_type}\n"
            summary_content += f"**Columns:** {', '.join(headers)}\n"
            summary_content += f"**Total Records:** {len(rows)}\n\n"
            
            # Add sample data
            summary_content += "## Data Summary\n\n"
            for row in rows[:20]:  # First 20 rows as summary
                row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                summary_content += f"- {row_text}\n"
            
            documents.append(Document(
                page_content=summary_content,
                metadata={
                    "source": str(filepath),
                    "filename": filename,
                    "company_id": company_id,
                    "doc_type": doc_type,
                    "category": "financial" if doc_type != "employee_data" else "hr",
                    "record_count": len(rows),
                }
            ))
            
            # For employee data, create individual employee records
            if doc_type == "employee_data":
                for i, row in enumerate(rows):
                    emp_content = f"Employee Record: {row.get('Employee_Name', 'Unknown')}\n"
                    for k, v in row.items():
                        if v and v != '--':
                            emp_content += f"- {k}: {v}\n"
                    
                    documents.append(Document(
                        page_content=emp_content,
                        metadata={
                            "source": str(filepath),
                            "filename": filename,
                            "company_id": row.get('Company', company_id),
                            "doc_type": "employee_record",
                            "category": "hr",
                            "employee_id": row.get('EmpID', str(i)),
                            "department": row.get('Department', 'Unknown'),
                            "position": row.get('Position', 'Unknown'),
                        }
                    ))
            
    except Exception as e:
        logger.error(f"Error reading CSV {filepath}: {e}")
    
    return documents


def read_markdown_file(filepath: Path) -> list[Document]:
    """Read markdown file and convert to documents."""
    documents = []
    filename = filepath.name
    company_id = identify_company(filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Skip filepath comment if present
        if content.startswith('<!--'):
            content = content.split('-->', 1)[-1].strip()
        
        # Determine document type
        doc_type = "policy_document"
        category = "hr"
        
        filename_lower = filename.lower()
        if "handbook" in filename_lower or "employee" in filename_lower:
            doc_type = "employee_handbook"
        elif "license" in filename_lower:
            doc_type = "license_agreement"
            category = "legal"
        elif "partnership" in filename_lower:
            doc_type = "partnership_agreement"
            category = "legal"
        elif "contract" in filename_lower or "agreement" in filename_lower:
            doc_type = "contract"
            category = "legal"
        elif "mnda" in filename_lower or "nda" in filename_lower:
            doc_type = "nda"
            category = "legal"
        elif "environment" in filename_lower:
            doc_type = "environmental_policy"
            category = "legal"
        elif "engagement" in filename_lower:
            doc_type = "engagement_agreement"
            category = "legal"
        
        documents.append(Document(
            page_content=content,
            metadata={
                "source": str(filepath),
                "filename": filename,
                "company_id": company_id,
                "doc_type": doc_type,
                "category": category,
            }
        ))
        
    except Exception as e:
        logger.error(f"Error reading MD {filepath}: {e}")
    
    return documents


def read_text_file(filepath: Path) -> list[Document]:
    """Read text file and convert to documents."""
    documents = []
    filename = filepath.name
    company_id = identify_company(filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Skip filepath comment if present
        if content.startswith('//'):
            content = content.split('\n', 1)[-1].strip()
        
        # Determine category and type based on parent folders
        parent_path = str(filepath.parent).lower()
        category = "legal"
        doc_type = "legal_document"
        
        if "litigation" in parent_path:
            doc_type = "litigation"
        elif "compliance" in parent_path:
            doc_type = "compliance"
        elif "contract" in parent_path:
            doc_type = "contract"
        elif "ip" in parent_path or "patent" in parent_path:
            doc_type = "intellectual_property"
        elif "court" in parent_path or "judgment" in parent_path:
            doc_type = "court_judgment"
        
        documents.append(Document(
            page_content=content,
            metadata={
                "source": str(filepath),
                "filename": filename,
                "company_id": company_id,
                "doc_type": doc_type,
                "category": category,
            }
        ))
        
    except Exception as e:
        logger.error(f"Error reading TXT {filepath}: {e}")
    
    return documents


def scan_directory_for_documents(base_path: Path) -> list[Document]:
    """Recursively scan directory for documents."""
    all_documents = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            filepath = Path(root) / file
            file_lower = file.lower()
            
            try:
                if file_lower.endswith('.csv'):
                    docs = read_csv_file(filepath)
                    all_documents.extend(docs)
                elif file_lower.endswith('.md'):
                    docs = read_markdown_file(filepath)
                    all_documents.extend(docs)
                elif file_lower.endswith('.txt'):
                    docs = read_text_file(filepath)
                    all_documents.extend(docs)
                # Skip PDFs for now (would need pypdf or similar)
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
    
    return all_documents


def load_financial_documents() -> int:
    """Load financial documents into ChromaDB."""
    logger.info("Loading financial documents...")
    
    finance_path = RAW_DATA_PATH / "Finance"
    if not finance_path.exists():
        logger.warning(f"Finance path not found: {finance_path}")
        return 0
    
    documents = scan_directory_for_documents(finance_path)
    
    if not documents:
        logger.warning("No financial documents found")
        return 0
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    split_docs = splitter.split_documents(documents)
    
    # Add to vector store
    vectorstore = get_vectorstore(COLLECTIONS["financial"])
    vectorstore.add_documents(split_docs)
    
    logger.info(f"Loaded {len(split_docs)} financial document chunks")
    return len(split_docs)


def load_legal_documents() -> int:
    """Load legal documents into ChromaDB."""
    logger.info("Loading legal documents...")
    
    legal_path = RAW_DATA_PATH / "legal"
    if not legal_path.exists():
        logger.warning(f"Legal path not found: {legal_path}")
        return 0
    
    documents = scan_directory_for_documents(legal_path)
    
    if not documents:
        logger.warning("No legal documents found")
        return 0
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    split_docs = splitter.split_documents(documents)
    
    # Add to vector store
    vectorstore = get_vectorstore(COLLECTIONS["legal"])
    vectorstore.add_documents(split_docs)
    
    logger.info(f"Loaded {len(split_docs)} legal document chunks")
    return len(split_docs)


def load_hr_documents() -> int:
    """Load HR documents into ChromaDB."""
    logger.info("Loading HR documents...")
    
    hr_path = RAW_DATA_PATH / "HR Data"
    if not hr_path.exists():
        logger.warning(f"HR path not found: {hr_path}")
        return 0
    
    documents = scan_directory_for_documents(hr_path)
    
    if not documents:
        logger.warning("No HR documents found")
        return 0
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    split_docs = splitter.split_documents(documents)
    
    # Add to vector store
    vectorstore = get_vectorstore(COLLECTIONS["hr"])
    vectorstore.add_documents(split_docs)
    
    logger.info(f"Loaded {len(split_docs)} HR document chunks")
    return len(split_docs)


def load_all_documents() -> dict[str, int]:
    """Load all documents into their respective vector stores."""
    results = {
        "financial": load_financial_documents(),
        "legal": load_legal_documents(),
        "hr": load_hr_documents(),
    }
    
    results["total"] = sum(results.values())
    
    logger.info(f"Total documents loaded: {results['total']}")
    return results


class DocumentLoader:
    """Class wrapper for document loading functions."""
    
    def __init__(self, data_dir: str = None, chroma_dir: str = None):
        """Initialize the document loader."""
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_PATH
        self.chroma_dir = chroma_dir
    
    def load_financial_documents(self) -> int:
        return load_financial_documents()
    
    def load_legal_documents(self) -> int:
        return load_legal_documents()
    
    def load_hr_documents(self) -> int:
        return load_hr_documents()
    
    def load_all_documents(self) -> dict[str, int]:
        return load_all_documents()
    
    @staticmethod
    def get_vectorstore(collection: str) -> Chroma:
        return get_vectorstore(COLLECTIONS.get(collection, collection))
    
    @staticmethod
    def search_documents(query: str, collection: str = "financial", k: int = 5) -> list[Document]:
        store = get_vectorstore(COLLECTIONS.get(collection, collection))
        return store.similarity_search(query, k=k)
    
    @staticmethod
    def search_with_filter(
        query: str,
        collection: str,
        company_id: str = None,
        doc_type: str = None,
        k: int = 5
    ) -> list[Document]:
        """Search with metadata filters."""
        store = get_vectorstore(COLLECTIONS.get(collection, collection))
        
        filter_dict = {}
        if company_id:
            filter_dict["company_id"] = company_id
        if doc_type:
            filter_dict["doc_type"] = doc_type
        
        if filter_dict:
            return store.similarity_search(query, k=k, filter=filter_dict)
        return store.similarity_search(query, k=k)


# Convenience function for backward compatibility
def load_all_data_to_vectorstore(data_path: str = None) -> dict[str, int]:
    """Load all data to vector store."""
    return load_all_documents()


if __name__ == "__main__":
    print("Loading documents into ChromaDB...")
    results = load_all_documents()
    print(f"\nResults:")
    for category, count in results.items():
        print(f"  {category}: {count} chunks")
