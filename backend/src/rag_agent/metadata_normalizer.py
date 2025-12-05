"""
Metadata Normalization for RAG Pipeline.

Ensures consistent metadata values across different ingestion paths.
Maps string values to canonical DocumentCategory and DocumentType enums.
"""

from typing import Dict, Any, Optional, Tuple
from src.rag_agent.base import DocumentCategory, DocumentType
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Mapping from various string values to canonical DocumentType enum
DOC_TYPE_MAPPINGS: Dict[str, DocumentType] = {
    # Financial types
    "balance_sheet": DocumentType.BALANCE_SHEET,
    "balancesheet": DocumentType.BALANCE_SHEET,
    "income_statement": DocumentType.INCOME_STATEMENT,
    "incomestatement": DocumentType.INCOME_STATEMENT,
    "cash_flow": DocumentType.CASH_FLOW,
    "cash_flow_statement": DocumentType.CASH_FLOW,
    "cashflow": DocumentType.CASH_FLOW,
    "cashflow_statement": DocumentType.CASH_FLOW,
    "financial_statement": DocumentType.FINANCIAL_STATEMENT,
    "financial_data": DocumentType.FINANCIAL_STATEMENT,
    
    # Legal types
    "contract": DocumentType.CONTRACT,
    "contracts": DocumentType.CONTRACT,
    "litigation": DocumentType.LITIGATION,
    "lawsuit": DocumentType.LITIGATION,
    "ip_document": DocumentType.IP_DOCUMENT,
    "intellectual_property": DocumentType.IP_DOCUMENT,
    "patent": DocumentType.IP_DOCUMENT,
    "trademark": DocumentType.IP_DOCUMENT,
    "copyright": DocumentType.IP_DOCUMENT,
    "compliance": DocumentType.COMPLIANCE,
    "regulatory": DocumentType.COMPLIANCE,
    "court_judgment": DocumentType.COURT_JUDGMENT,
    "judgment": DocumentType.COURT_JUDGMENT,
    "nda": DocumentType.NDA,
    "mnda": DocumentType.NDA,
    "license_agreement": DocumentType.LICENSE_AGREEMENT,
    "license": DocumentType.LICENSE_AGREEMENT,
    "partnership_agreement": DocumentType.PARTNERSHIP_AGREEMENT,
    "partnership": DocumentType.PARTNERSHIP_AGREEMENT,
    "environmental_policy": DocumentType.ENVIRONMENTAL_POLICY,
    "environment": DocumentType.ENVIRONMENTAL_POLICY,
    
    # HR types
    "employee_record": DocumentType.EMPLOYEE_RECORD,
    "employee_data": DocumentType.EMPLOYEE_RECORD,
    "hr_policy": DocumentType.HR_POLICY,
    "policy_document": DocumentType.POLICY_DOCUMENT,
    "employee_handbook": DocumentType.EMPLOYEE_HANDBOOK,
    "handbook": DocumentType.EMPLOYEE_HANDBOOK,
    
    # Generic/Unknown
    "legal_document": DocumentType.UNKNOWN,
    "pdf_document": DocumentType.UNKNOWN,
    "engagement_agreement": DocumentType.CONTRACT,  # Map to closest type
    "unknown": DocumentType.UNKNOWN,
}

# Mapping from various string values to canonical DocumentCategory enum
CATEGORY_MAPPINGS: Dict[str, DocumentCategory] = {
    "financial": DocumentCategory.FINANCIAL,
    "finance": DocumentCategory.FINANCIAL,
    "legal": DocumentCategory.LEGAL,
    "hr": DocumentCategory.HR,
    "human_resources": DocumentCategory.HR,
    "market": DocumentCategory.MARKET,
    "cross_ref": DocumentCategory.CROSS_REF,
    "unknown": DocumentCategory.UNKNOWN,
}


def normalize_doc_type(doc_type_value: Any) -> DocumentType:
    """
    Normalize a document type value to canonical DocumentType enum.
    
    Args:
        doc_type_value: String or DocumentType enum
        
    Returns:
        Canonical DocumentType enum value
    """
    if isinstance(doc_type_value, DocumentType):
        return doc_type_value
    
    if doc_type_value is None:
        return DocumentType.UNKNOWN
    
    # Normalize string value
    normalized = str(doc_type_value).lower().strip().replace("-", "_").replace(" ", "_")
    
    # Check direct mapping
    if normalized in DOC_TYPE_MAPPINGS:
        return DOC_TYPE_MAPPINGS[normalized]
    
    # Try to match enum value directly
    try:
        return DocumentType(normalized)
    except ValueError:
        pass
    
    # Log unmapped value for debugging
    logger.debug(f"Unmapped doc_type value: {doc_type_value} -> UNKNOWN")
    return DocumentType.UNKNOWN


def normalize_category(category_value: Any) -> DocumentCategory:
    """
    Normalize a category value to canonical DocumentCategory enum.
    
    Args:
        category_value: String or DocumentCategory enum
        
    Returns:
        Canonical DocumentCategory enum value
    """
    if isinstance(category_value, DocumentCategory):
        return category_value
    
    if category_value is None:
        return DocumentCategory.UNKNOWN
    
    # Normalize string value
    normalized = str(category_value).lower().strip().replace("-", "_").replace(" ", "_")
    
    # Check direct mapping
    if normalized in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[normalized]
    
    # Try to match enum value directly
    try:
        return DocumentCategory(normalized)
    except ValueError:
        pass
    
    logger.debug(f"Unmapped category value: {category_value} -> UNKNOWN")
    return DocumentCategory.UNKNOWN


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metadata dictionary to use canonical enum values.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Normalized metadata dictionary with enum values as strings
    """
    normalized = metadata.copy()
    
    # Normalize doc_type
    if "doc_type" in normalized:
        original = normalized["doc_type"]
        doc_type_enum = normalize_doc_type(original)
        normalized["doc_type"] = doc_type_enum.value
        if original != doc_type_enum.value:
            normalized["original_doc_type"] = str(original)
    
    # Normalize category
    if "category" in normalized:
        original = normalized["category"]
        category_enum = normalize_category(original)
        normalized["category"] = category_enum.value
        if original != category_enum.value:
            normalized["original_category"] = str(original)
    
    return normalized


def get_canonical_values(doc_type_str: Optional[str], category_str: Optional[str]) -> Tuple[DocumentType, DocumentCategory]:
    """
    Get canonical enum values from string inputs.
    
    Args:
        doc_type_str: Document type string
        category_str: Category string
        
    Returns:
        Tuple of (DocumentType, DocumentCategory)
    """
    doc_type = normalize_doc_type(doc_type_str) if doc_type_str else DocumentType.UNKNOWN
    category = normalize_category(category_str) if category_str else DocumentCategory.UNKNOWN
    return doc_type, category
