"""Data module for synthetic data generation and management."""

# Import synthetic data generator components (always available)
from data.synthetic_data_generator import (
    SyntheticDataGenerator,
    generate_all_synthetic_data,
    generate_financial_statements,
    generate_legal_documents,
    generate_hr_data,
    COMPANIES,
)

# Import document loader components (requires langchain dependencies)
try:
    from data.document_loader import (
        DocumentLoader,
        load_all_data_to_vectorstore,
        get_vectorstore,
        COLLECTIONS,
    )
except ImportError:
    # langchain dependencies not installed yet
    DocumentLoader = None
    load_all_data_to_vectorstore = None
    get_vectorstore = None
    COLLECTIONS = {}

__all__ = [
    # Classes
    "SyntheticDataGenerator",
    "DocumentLoader",
    # Generator functions
    "generate_all_synthetic_data",
    "generate_financial_statements",
    "generate_legal_documents",
    "generate_hr_data",
    # Loader functions
    "load_all_data_to_vectorstore",
    "get_vectorstore",
    # Constants
    "COMPANIES",
    "COLLECTIONS",
]
