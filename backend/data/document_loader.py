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
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from src.config.llm_config import get_embedding_model
from src.config.settings import get_settings
from src.common.logging_config import get_logger
from src.rag_agent.metadata_normalizer import normalize_metadata, normalize_doc_type, normalize_category
from src.rag_agent.base import DocumentType, DocumentCategory

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
            
            # Determine document type from filename - use canonical enum values
            doc_type = DocumentType.FINANCIAL_STATEMENT.value  # default
            category = DocumentCategory.FINANCIAL.value
            is_financial = True
            
            if "balance" in filename.lower():
                doc_type = DocumentType.BALANCE_SHEET.value
            elif "income" in filename.lower():
                doc_type = DocumentType.INCOME_STATEMENT.value
            elif "cashflow" in filename.lower() or "cash_flow" in filename.lower():
                doc_type = DocumentType.CASH_FLOW.value
            elif "employee" in filename.lower():
                doc_type = DocumentType.EMPLOYEE_RECORD.value
                category = DocumentCategory.HR.value
                is_financial = False
            
            headers = list(rows[0].keys()) if rows else []
            
            # For FINANCIAL documents: Create COMPLETE data document with ALL rows
            if is_financial:
                # Create comprehensive document with FULL data
                full_content = f"# {filename} - Complete Financial Data\n\n"
                full_content += f"**Company:** {company_id}\n"
                full_content += f"**Document Type:** {doc_type}\n"
                full_content += f"**Columns:** {', '.join(headers)}\n"
                full_content += f"**Total Records:** {len(rows)}\n\n"
                full_content += "## Complete Financial Data\n\n"
                
                # Add header row as table
                full_content += "| " + " | ".join(headers) + " |\n"
                full_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                
                # Add ALL data rows - this is the key fix!
                for row in rows:
                    row_values = [str(row.get(h, "")) for h in headers]
                    full_content += "| " + " | ".join(row_values) + " |\n"
                
                documents.append(Document(
                    page_content=full_content,
                    metadata={
                        "source": str(filepath),
                        "filename": filename,
                        "company_id": company_id,
                        "doc_type": doc_type,
                        "category": category,
                        "record_count": len(rows),
                        "data_complete": True,
                    }
                ))
                
                # Also create individual row chunks for granular search
                for i, row in enumerate(rows):
                    row_label = row.get(headers[0], f"Row {i+1}") if headers else f"Row {i+1}"
                    
                    row_content = f"# {filename} - {row_label}\n\n"
                    row_content += f"**Company:** {company_id}\n"
                    row_content += f"**Document Type:** {doc_type}\n"
                    row_content += f"**Financial Metric:** {row_label}\n\n"
                    row_content += "## Values\n\n"
                    
                    for header in headers:
                        value = row.get(header, "")
                        if value and value != "--":
                            row_content += f"- **{header}:** {value}\n"
                    
                    documents.append(Document(
                        page_content=row_content,
                        metadata={
                            "source": str(filepath),
                            "filename": filename,
                            "company_id": company_id,
                            "doc_type": doc_type,
                            "category": category,
                            "row_index": i,
                            "row_label": row_label,
                        }
                    ))
            
            # For EMPLOYEE data: Create individual employee records
            elif doc_type == DocumentType.EMPLOYEE_RECORD.value:
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
                            "doc_type": DocumentType.EMPLOYEE_RECORD.value,
                            "category": DocumentCategory.HR.value,
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
        
        # Determine document type - use canonical enum values
        doc_type = DocumentType.POLICY_DOCUMENT.value
        category = DocumentCategory.HR.value
        
        filename_lower = filename.lower()
        if "handbook" in filename_lower or "employee" in filename_lower:
            doc_type = DocumentType.EMPLOYEE_HANDBOOK.value
        elif "license" in filename_lower:
            doc_type = DocumentType.LICENSE_AGREEMENT.value
            category = DocumentCategory.LEGAL.value
        elif "partnership" in filename_lower:
            doc_type = DocumentType.PARTNERSHIP_AGREEMENT.value
            category = DocumentCategory.LEGAL.value
        elif "contract" in filename_lower or "agreement" in filename_lower:
            doc_type = DocumentType.CONTRACT.value
            category = DocumentCategory.LEGAL.value
        elif "mnda" in filename_lower or "nda" in filename_lower:
            doc_type = DocumentType.NDA.value
            category = DocumentCategory.LEGAL.value
        elif "environment" in filename_lower:
            doc_type = DocumentType.ENVIRONMENTAL_POLICY.value
            category = DocumentCategory.LEGAL.value
        elif "engagement" in filename_lower:
            doc_type = DocumentType.CONTRACT.value  # Map engagement to contract
            category = DocumentCategory.LEGAL.value
        
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
        
        # Determine category and type based on parent folders - use canonical enum values
        parent_path = str(filepath.parent).lower()
        category = DocumentCategory.LEGAL.value
        doc_type = DocumentType.UNKNOWN.value
        
        if "litigation" in parent_path:
            doc_type = DocumentType.LITIGATION.value
        elif "compliance" in parent_path:
            doc_type = DocumentType.COMPLIANCE.value
        elif "contract" in parent_path:
            doc_type = DocumentType.CONTRACT.value
        elif "ip" in parent_path or "patent" in parent_path:
            doc_type = DocumentType.IP_DOCUMENT.value
        elif "court" in parent_path or "judgment" in parent_path:
            doc_type = DocumentType.COURT_JUDGMENT.value
        
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


def read_pdf_file(filepath: Path) -> list[Document]:
    """
    Read PDF file and convert to documents.
    
    Handles:
    - Native PDFs (text extraction via pypdf)
    - Scanned PDFs (OCR via pytesseract as fallback)
    - Multi-page documents
    
    Args:
        filepath: Path to PDF file
        
    Returns:
        List of Document objects with metadata
    """
    from pypdf import PdfReader
    
    documents = []
    filename = filepath.name
    
    try:
        # Identify company from filepath
        company_id = identify_company(str(filepath))
        
        # Determine category based on parent folders - use canonical enum values
        parent_path = str(filepath.parent).lower()
        category = DocumentCategory.LEGAL.value
        doc_type = DocumentType.UNKNOWN.value
        
        if "litigation" in parent_path:
            doc_type = DocumentType.LITIGATION.value
        elif "compliance" in parent_path:
            doc_type = DocumentType.COMPLIANCE.value
        elif "contract" in parent_path:
            doc_type = DocumentType.CONTRACT.value
        elif "ip" in parent_path or "patent" in parent_path:
            doc_type = DocumentType.IP_DOCUMENT.value
        elif "court" in parent_path or "judgment" in parent_path:
            doc_type = DocumentType.COURT_JUDGMENT.value
        
        # Try native PDF text extraction first
        reader = PdfReader(filepath)
        text_content = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_content.append(f"[Page {page_num + 1}]\n{page_text}")
        
        full_text = "\n\n".join(text_content)
        
        # If no text extracted, attempt OCR
        if not full_text.strip():
            logger.info(f"No text in PDF, attempting OCR: {filename}")
            full_text = _extract_text_via_ocr(filepath)
        
        if full_text.strip():
            documents.append(Document(
                page_content=full_text,
                metadata={
                    "source": str(filepath),
                    "filename": filename,
                    "company_id": company_id,
                    "doc_type": doc_type,
                    "category": category,
                    "page_count": len(reader.pages),
                }
            ))
            logger.info(f"Loaded PDF: {filename} ({len(reader.pages)} pages, {len(full_text)} chars)")
        else:
            logger.warning(f"No text extracted from PDF: {filename}")
            
    except Exception as e:
        logger.error(f"Error reading PDF {filepath}: {e}")
    
    return documents


def _extract_text_via_ocr(filepath: Path) -> str:
    """
    Extract text from scanned PDF using OCR.
    
    Uses PyMuPDF (fitz) to render pages to images, then Tesseract for OCR.
    This approach doesn't require Poppler.
    
    Args:
        filepath: Path to PDF file
        
    Returns:
        Extracted text string
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import io
        from src.config.settings import get_settings
        
        # Configure Tesseract path
        settings = get_settings()
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        
        doc = fitz.open(str(filepath))
        text_parts = []
        
        for page_num in range(min(len(doc), 50)):  # Limit to 50 pages for performance
            page = doc.load_page(page_num)
            # Render page to image at 150 DPI for good OCR quality
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Run OCR on the page image
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        return "\n\n".join(text_parts)
        
    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {e}")
        return ""
    except Exception as e:
        logger.error(f"OCR failed for {filepath}: {e}")
        return ""


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
                elif file_lower.endswith('.pdf'):
                    docs = read_pdf_file(filepath)
                    all_documents.extend(docs)
                
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
