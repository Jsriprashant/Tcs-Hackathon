"""
Document loaders for different file types.

Supports CSV, TXT, MD, PDF (text + OCR), and images.
"""

import csv
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_core.documents import Document

from src.rag_agent.base import (
    BaseLoader, DocumentChunk, ChunkMetadata,
    DocumentCategory, DocumentType
)
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Company mappings (same as existing)
COMPANIES = {
    "BBD": {"name": "BBD Ltd", "aliases": ["BBD_LTD", "BBD Software", "BBD_Software"]},
    "XYZ": {"name": "XYZ Ltd", "aliases": ["XYZ_LTD", "XYZ LTD"]},
    "SUPERNOVA": {"name": "Supernova Inc", "aliases": ["Supernova", "Supernova_"]},
    "RASPUTIN": {"name": "Rasputin Petroleum Ltd", "aliases": ["Rasputin", "Rasputil_Petroleum", "Rasputin_"]},
    "TECHNOBOX": {"name": "Techno Box Inc", "aliases": ["Techno_Box", "TechnoBox"]},
}


def identify_company(filename: str, content: str = "") -> str:
    """Identify company from filename and content."""
    text_to_check = (filename + " " + content[:1000]).upper()
    
    for company_id, info in COMPANIES.items():
        if company_id in text_to_check:
            return company_id
        for alias in info["aliases"]:
            if alias.upper() in text_to_check:
                return company_id
    
    return "UNKNOWN"


def infer_category_and_type(filepath: Path, content: str = "") -> tuple[DocumentCategory, DocumentType]:
    """Infer document category and type from path and content."""
    path_str = str(filepath).lower()
    filename = filepath.name.lower()
    content_lower = content[:500].lower()
    
    # Category inference
    category = DocumentCategory.UNKNOWN
    if "finance" in path_str or "financial" in path_str:
        category = DocumentCategory.FINANCIAL
    elif "legal" in path_str or "contract" in path_str or "litigation" in path_str:
        category = DocumentCategory.LEGAL
    elif "hr" in path_str or "employee" in path_str or "human" in path_str:
        category = DocumentCategory.HR
    elif "market" in path_str:
        category = DocumentCategory.MARKET
    
    # Document type inference
    doc_type = DocumentType.UNKNOWN
    
    # Financial types
    if "balance" in filename or "balancesheet" in filename:
        doc_type = DocumentType.BALANCE_SHEET
        category = DocumentCategory.FINANCIAL
    elif "income" in filename or "incomestatement" in filename:
        doc_type = DocumentType.INCOME_STATEMENT
        category = DocumentCategory.FINANCIAL
    elif "cash" in filename and "flow" in filename:
        doc_type = DocumentType.CASH_FLOW
        category = DocumentCategory.FINANCIAL
    
    # Legal types
    elif "contract" in filename or "agreement" in filename:
        doc_type = DocumentType.CONTRACT
        category = DocumentCategory.LEGAL
    elif "litigation" in path_str or "lawsuit" in filename:
        doc_type = DocumentType.LITIGATION
        category = DocumentCategory.LEGAL
    elif "patent" in path_str or "trademark" in path_str or "copyright" in path_str:
        doc_type = DocumentType.IP_DOCUMENT
        category = DocumentCategory.LEGAL
    elif "compliance" in path_str or "regulatory" in filename:
        doc_type = DocumentType.COMPLIANCE
        category = DocumentCategory.LEGAL
    elif "judgment" in filename or "court" in path_str:
        doc_type = DocumentType.COURT_JUDGMENT
        category = DocumentCategory.LEGAL
    elif "nda" in filename or "mnda" in filename:
        doc_type = DocumentType.NDA
        category = DocumentCategory.LEGAL
    elif "license" in filename:
        doc_type = DocumentType.LICENSE_AGREEMENT
        category = DocumentCategory.LEGAL
    elif "partnership" in filename:
        doc_type = DocumentType.PARTNERSHIP_AGREEMENT
        category = DocumentCategory.LEGAL
    elif "environment" in filename:
        doc_type = DocumentType.ENVIRONMENTAL_POLICY
        category = DocumentCategory.LEGAL
    
    # HR types
    elif "employee" in filename and "data" in filename:
        doc_type = DocumentType.EMPLOYEE_RECORD
        category = DocumentCategory.HR
    elif "handbook" in filename:
        doc_type = DocumentType.EMPLOYEE_HANDBOOK
        category = DocumentCategory.HR
    elif "policy" in filename or "policies" in path_str:
        doc_type = DocumentType.HR_POLICY
        category = DocumentCategory.HR
    
    return category, doc_type


class CSVLoader(BaseLoader):
    """Loader for CSV files."""
    
    # Financial document types that need full data ingestion
    FINANCIAL_DOC_TYPES = {
        DocumentType.BALANCE_SHEET,
        DocumentType.INCOME_STATEMENT,
        DocumentType.CASH_FLOW,
        DocumentType.FINANCIAL_STATEMENT,
    }
    
    def __init__(self, create_individual_records: bool = True):
        """
        Initialize CSV loader.
        
        Args:
            create_individual_records: If True, create chunks for each row
        """
        self.create_individual_records = create_individual_records
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is CSV."""
        return file_path.suffix.lower() == '.csv'
    
    def _create_financial_chunks(
        self, 
        rows: List[Dict], 
        headers: List[str],
        filename: str,
        file_path: Path,
        category: DocumentCategory,
        doc_type: DocumentType,
        company_id: str
    ) -> List[DocumentChunk]:
        """Create comprehensive chunks for financial documents with FULL data."""
        chunks = []
        
        # Create a COMPLETE data chunk with all rows (primary chunk for retrieval)
        full_content = f"# {filename} - Complete Financial Data\n\n"
        full_content += f"**Company:** {company_id}\n"
        full_content += f"**Document Type:** {doc_type.value}\n"
        full_content += f"**Columns:** {', '.join(headers)}\n"
        full_content += f"**Total Records:** {len(rows)}\n\n"
        full_content += "## Complete Data\n\n"
        
        # Add header row
        full_content += "| " + " | ".join(headers) + " |\n"
        full_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        # Add ALL data rows in table format
        for row in rows:
            row_values = [str(row.get(h, "")) for h in headers]
            full_content += "| " + " | ".join(row_values) + " |\n"
        
        full_metadata = ChunkMetadata(
            source=str(file_path),
            filename=filename,
            company_id=company_id,
            category=category,
            doc_type=doc_type,
            chunk_hash="",
            chunk_index=0,
            total_chunks=1,
            record_count=len(rows),
        )
        
        chunks.append(DocumentChunk(
            content=full_content,
            metadata=full_metadata
        ))
        
        # Create individual row chunks for granular retrieval
        for i, row in enumerate(rows):
            row_label = row.get(headers[0], f"Row {i+1}") if headers else f"Row {i+1}"
            
            row_content = f"# {filename} - {row_label}\n\n"
            row_content += f"**Company:** {company_id}\n"
            row_content += f"**Document Type:** {doc_type.value}\n"
            row_content += f"**Row:** {i+1} of {len(rows)}\n\n"
            row_content += "## Data\n\n"
            
            for header in headers:
                value = row.get(header, "")
                if value and value != "--":
                    row_content += f"- **{header}:** {value}\n"
            
            row_metadata = ChunkMetadata(
                source=str(file_path),
                filename=filename,
                company_id=company_id,
                category=category,
                doc_type=doc_type,
                chunk_hash="",
                chunk_index=i + 1,
                total_chunks=len(rows) + 1,
                record_count=1,
            )
            
            chunks.append(DocumentChunk(
                content=row_content,
                metadata=row_metadata
            ))
        
        return chunks
    
    def _create_employee_chunks(
        self,
        rows: List[Dict],
        filename: str,
        file_path: Path,
        company_id: str
    ) -> List[DocumentChunk]:
        """Create chunks for employee/HR data."""
        chunks = []
        
        for i, row in enumerate(rows):
            emp_content = f"# Employee Record\n\n"
            emp_content += f"**Source:** {filename}\n"
            emp_content += f"**Record:** {i+1} of {len(rows)}\n\n"
            emp_content += "## Employee Details\n\n"
            
            for k, v in row.items():
                if v and v != '--':
                    emp_content += f"- **{k}:** {v}\n"
            
            emp_metadata = ChunkMetadata(
                source=str(file_path),
                filename=filename,
                company_id=row.get('Company', company_id),
                category=DocumentCategory.HR,
                doc_type=DocumentType.EMPLOYEE_RECORD,
                chunk_hash="",
                chunk_index=i,
                total_chunks=len(rows),
            )
            
            chunks.append(DocumentChunk(
                content=emp_content,
                metadata=emp_metadata
            ))
        
        return chunks
    
    def load(self, file_path: Path) -> List[DocumentChunk]:
        """Load CSV file into document chunks."""
        chunks = []
        filename = file_path.name
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip comment line if present
                first_line = f.readline()
                if not first_line.startswith('//'):
                    f.seek(0)
                
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return chunks
                
                # Infer category and type
                category, doc_type = infer_category_and_type(file_path)
                company_id = identify_company(filename)
                headers = list(rows[0].keys()) if rows else []
                
                logger.info(f"Loading {filename}: {len(rows)} rows, type={doc_type.value}, company={company_id}")
                
                # Handle financial documents - create full data chunks
                if doc_type in self.FINANCIAL_DOC_TYPES or category == DocumentCategory.FINANCIAL:
                    chunks = self._create_financial_chunks(
                        rows, headers, filename, file_path, 
                        category, doc_type, company_id
                    )
                    logger.info(f"Created {len(chunks)} financial chunks for {filename}")
                
                # Handle employee/HR data
                elif doc_type == DocumentType.EMPLOYEE_RECORD:
                    chunks = self._create_employee_chunks(
                        rows, filename, file_path, company_id
                    )
                    logger.info(f"Created {len(chunks)} employee chunks for {filename}")
                
                # Handle other CSV types - create comprehensive chunk
                else:
                    # Create complete data chunk
                    full_content = f"# {filename}\n\n"
                    full_content += f"**Company:** {company_id}\n"
                    full_content += f"**Category:** {category.value}\n"
                    full_content += f"**Document Type:** {doc_type.value}\n"
                    full_content += f"**Columns:** {', '.join(headers)}\n"
                    full_content += f"**Total Records:** {len(rows)}\n\n"
                    full_content += "## Complete Data\n\n"
                    
                    # Add all rows
                    for row in rows:
                        row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v and v != '--'])
                        full_content += f"- {row_text}\n"
                    
                    metadata = ChunkMetadata(
                        source=str(file_path),
                        filename=filename,
                        company_id=company_id,
                        category=category,
                        doc_type=doc_type,
                        chunk_hash="",
                        record_count=len(rows),
                    )
                    
                    chunks.append(DocumentChunk(
                        content=full_content,
                        metadata=metadata
                    ))
        
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
        
        return chunks


class TextLoader(BaseLoader):
    """Loader for TXT and MD files."""
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is TXT or MD."""
        return file_path.suffix.lower() in ['.txt', '.md']
    
    def load(self, file_path: Path) -> List[DocumentChunk]:
        """Load text/markdown file."""
        chunks = []
        filename = file_path.name
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip comment lines
            if content.startswith('<!--'):
                content = content.split('-->', 1)[-1].strip()
            elif content.startswith('//'):
                content = content.split('\n', 1)[-1].strip()
            
            if not content.strip():
                return chunks
            
            # Infer metadata
            category, doc_type = infer_category_and_type(file_path, content)
            company_id = identify_company(filename, content)
            
            metadata = ChunkMetadata(
                source=str(file_path),
                filename=filename,
                company_id=company_id,
                category=category,
                doc_type=doc_type,
                chunk_hash="",
            )
            
            chunks.append(DocumentChunk(
                content=content,
                metadata=metadata
            ))
        
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
        
        return chunks


class PDFLoader(BaseLoader):
    """Loader for PDF files (text extraction + OCR fallback)."""
    
    def __init__(self, use_ocr: bool = True):
        """
        Initialize PDF loader.
        
        Args:
            use_ocr: If True, use OCR for scanned PDFs
        """
        self.use_ocr = use_ocr
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is PDF."""
        return file_path.suffix.lower() == '.pdf'
    
    def load(self, file_path: Path) -> List[DocumentChunk]:
        """Load PDF file with text extraction and OCR fallback."""
        chunks = []
        filename = file_path.name
        
        try:
            # Try text extraction first
            from pypdf import PdfReader
            
            reader = PdfReader(str(file_path))
            full_text = ""
            page_texts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    page_texts.append((page_num + 1, page_text))
                    full_text += page_text + "\n\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
            
            # Check if we got meaningful text
            if len(full_text.strip()) < 100 and self.use_ocr:
                logger.info(f"PDF {filename} appears scanned, attempting OCR...")
                return self._load_with_ocr(file_path)
            
            # Infer metadata
            category, doc_type = infer_category_and_type(file_path, full_text)
            company_id = identify_company(filename, full_text)
              # Create chunks per page or combined
            if len(page_texts) > 1:
                for page_num, page_text in page_texts:
                    if page_text.strip():
                        metadata = ChunkMetadata(
                            source=str(file_path),
                            filename=filename,
                            company_id=company_id,
                            category=category,
                            doc_type=doc_type,
                            chunk_hash="",
                            page=page_num,
                        )
                        
                        chunks.append(DocumentChunk(
                            content=page_text,
                            metadata=metadata
                        ))
            else:
                # Single chunk for small PDFs
                metadata = ChunkMetadata(
                    source=str(file_path),
                    filename=filename,
                    company_id=company_id,
                    category=category,
                    doc_type=doc_type,
                    chunk_hash="",
                )
                
                chunks.append(DocumentChunk(
                    content=full_text,
                    metadata=metadata
                ))
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            if self.use_ocr:
                logger.info("Attempting OCR fallback...")
                return self._load_with_ocr(file_path)
        
        return chunks
    
    def _load_with_ocr(self, file_path: Path) -> List[DocumentChunk]:
        """Load PDF using OCR with multiple fallback methods."""
        chunks = []
        filename = file_path.name
        
        # Try Method 1: PyMuPDF (fitz) for text extraction - doesn't require poppler
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(file_path))
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += text + "\n\n"
            
            # If we got text, return it
            if full_text.strip():
                doc.close()
                category, doc_type = infer_category_and_type(file_path, full_text)
                company_id = identify_company(filename, full_text)
                
                metadata = ChunkMetadata(
                    source=str(file_path),
                    filename=filename,
                    company_id=company_id,
                    category=category,
                    doc_type=doc_type,
                    chunk_hash="",
                )
                
                chunks.append(DocumentChunk(
                    content=full_text,
                    metadata=metadata
                ))
                logger.info(f"Successfully extracted text from {filename} using PyMuPDF")
                return chunks
            
            # If no text, try OCR using PyMuPDF to render images + pytesseract
            try:
                import pytesseract
                from PIL import Image
                import io
                from src.config.settings import get_settings
                
                # Configure Tesseract path
                settings = get_settings()
                if settings.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
                
                logger.info(f"PDF {filename} is scanned, using PyMuPDF + Tesseract OCR...")
                full_text = ""
                
                for page_num in range(min(len(doc), 50)):  # Limit to 50 pages for performance
                    page = doc.load_page(page_num)
                    # Render page to image at 150 DPI for good OCR quality
                    mat = fitz.Matrix(150/72, 150/72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # OCR the image
                    page_text = pytesseract.image_to_string(img)
                    if page_text.strip():
                        full_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                
                doc.close()
                
                if full_text.strip():
                    category, doc_type = infer_category_and_type(file_path, full_text)
                    company_id = identify_company(filename, full_text)
                    
                    metadata = ChunkMetadata(
                        source=str(file_path),
                        filename=filename,
                        company_id=company_id,
                        category=category,
                        doc_type=doc_type,
                        chunk_hash="",
                    )
                    
                    chunks.append(DocumentChunk(
                        content=full_text,
                        metadata=metadata
                    ))
                    logger.info(f"Successfully OCR'd {filename} using PyMuPDF + Tesseract")
                    return chunks
                    
            except ImportError as e:
                logger.warning(f"Tesseract OCR not available: {e}")
            except Exception as e:
                logger.warning(f"PyMuPDF + Tesseract OCR failed: {e}")
            
            doc.close()
                
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed.")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {file_path}: {e}")
          # Try Method 2: pdfplumber as fallback
        try:
            import pdfplumber
            
            full_text = ""
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
            
            if full_text.strip():
                category, doc_type = infer_category_and_type(file_path, full_text)
                company_id = identify_company(filename, full_text)
                
                metadata = ChunkMetadata(
                    source=str(file_path),
                    filename=filename,
                    company_id=company_id,
                    category=category,
                    doc_type=doc_type,
                    chunk_hash="",
                )
                
                chunks.append(DocumentChunk(
                    content=full_text,
                    metadata=metadata
                ))
                logger.info(f"Successfully extracted text from {filename} using pdfplumber")
                return chunks
                
        except ImportError:
            logger.debug("pdfplumber not installed.")
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed for {file_path}: {e}")
        
        # If all methods fail, create a placeholder chunk with metadata
        if not chunks:
            logger.warning(f"Could not extract text from PDF: {filename} (may be scanned/image-based)")
            category, doc_type = infer_category_and_type(file_path)
            company_id = identify_company(filename)
            
            # Create minimal metadata chunk so the document is at least indexed
            placeholder_content = f"# {filename}\n\n"
            placeholder_content += f"**Document Type:** PDF\n"
            placeholder_content += f"**Category:** {category.value}\n"
            placeholder_content += f"**Company:** {company_id}\n"
            placeholder_content += f"**Note:** Text extraction failed. Document may be scanned or image-based.\n"
            
            metadata = ChunkMetadata(
                source=str(file_path),
                filename=filename,
                company_id=company_id,
                category=category,
                doc_type=doc_type,
                chunk_hash="",
            )
            
            chunks.append(DocumentChunk(
                content=placeholder_content,
                metadata=metadata
            ))
        
        return chunks


class ImageLoader(BaseLoader):
    """Loader for images using OCR."""
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is an image."""
        return file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    def load(self, file_path: Path) -> List[DocumentChunk]:
        """Load image using OCR."""
        chunks = []
        
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            if text.strip():
                category, doc_type = infer_category_and_type(file_path, text)
                company_id = identify_company(file_path.name, text)
                
                metadata = ChunkMetadata(
                    source=str(file_path),
                    filename=file_path.name,
                    company_id=company_id,
                    category=category,
                    doc_type=doc_type,
                    chunk_hash="",
                )
                
                chunks.append(DocumentChunk(
                    content=text,
                    metadata=metadata
                ))
        
        except ImportError:
            logger.error("pytesseract not installed. Cannot perform OCR on images.")
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
        
        return chunks


def get_loader_for_file(file_path: Path) -> Optional[BaseLoader]:
    """Get appropriate loader for a file."""
    loaders = [
        CSVLoader(),
        TextLoader(),
        PDFLoader(),
        ImageLoader(),
    ]
    
    for loader in loaders:
        if loader.supports(file_path):
            return loader
    
    return None
