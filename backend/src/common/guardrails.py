"""Security guardrails for M&A Due Diligence Platform."""

import re
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class PIIPattern:
    """Pattern for detecting PII."""
    name: str
    pattern: str
    replacement: str


class PIIFilter:
    """
    Filter for detecting and redacting Personally Identifiable Information.
    
    Prevents sensitive data from being sent to LLM.
    """
    
    DEFAULT_PATTERNS = [
        PIIPattern("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
        PIIPattern("credit_card", r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CC_REDACTED]"),
        PIIPattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),        PIIPattern("phone", r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE_REDACTED]"),
        PIIPattern("pan_card", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[PAN_REDACTED]"),  # Indian PAN
        PIIPattern("aadhaar", r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[AADHAAR_REDACTED]"),  # Indian Aadhaar
        PIIPattern("bank_account", r"\b\d{9,18}\b", "[ACCOUNT_REDACTED]"),
    ]
    
    def __init__(self, additional_patterns: Optional[list[PIIPattern]] = None):
        self.patterns = self.DEFAULT_PATTERNS + (additional_patterns or [])
    
    def filter(self, text: str | list) -> tuple[str, list[str]]:
        """
        Filter PII from text.
        
        Args:
            text: Input text to filter (can be a string or list for multimodal content)
        
        Returns:
            Tuple of (filtered_text, list of detected PII types)
        """
        # Handle multimodal content (list of content blocks)
        if isinstance(text, list):
            # Extract text content from list
            text_parts = []
            for item in text:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            text = " ".join(text_parts)
        
        if not text:
            return "", []
        
        detected = []
        filtered_text = text
        
        for pii_pattern in self.patterns:
            if re.search(pii_pattern.pattern, filtered_text, re.IGNORECASE):
                detected.append(pii_pattern.name)
                filtered_text = re.sub(
                    pii_pattern.pattern, 
                    pii_pattern.replacement, 
                    filtered_text, 
                    flags=re.IGNORECASE
                )
        
        return filtered_text, detected
    
    def contains_pii(self, text: str | list) -> bool:
        """Check if text contains PII."""
        # Handle multimodal content (list of content blocks)
        if isinstance(text, list):
            text_parts = []
            for item in text:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            text = " ".join(text_parts)
        
        if not text:
            return False
        
        for pii_pattern in self.patterns:
            if re.search(pii_pattern.pattern, text, re.IGNORECASE):
                return True
        return False


class InputValidator:
    """
    Validate and sanitize user inputs.
    
    Ensures inputs meet security and format requirements.
    """
    
    MAX_QUERY_LENGTH = 10000
    BLOCKED_KEYWORDS = [
        "ignore previous instructions",
        "disregard all",
        "override system",
        "jailbreak",
        "bypass security",
    ]
    
    def validate_query(self, query: str | list) -> tuple[bool, Optional[str]]:
        """
        Validate a user query.
        
        Args:
            query: User input query (can be a string or list for multimodal content)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle multimodal content (list of content blocks)
        if isinstance(query, list):
            # Extract text content from list
            text_parts = []
            for item in query:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            query = " ".join(text_parts)
        
        if not query:
            return False, "Query cannot be empty"
        
        if len(query) > self.MAX_QUERY_LENGTH:
            return False, f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH}"
        
        query_lower = query.lower()
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in query_lower:
                return False, "Query contains blocked content"
        
        return True, None
    
    def validate_company_id(self, company_id: str) -> tuple[bool, Optional[str]]:
        """Validate company identifier format."""
        if not company_id:
            return False, "Company ID cannot be empty"
        
        # Allow alphanumeric with underscores/hyphens
        if not re.match(r"^[A-Za-z0-9_-]+$", company_id):
            return False, "Invalid company ID format"
        
        return True, None
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal."""
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[\\/:*?"<>|]', '_', filename)
        # Remove any path traversal attempts
        sanitized = sanitized.replace('..', '')
        return sanitized


class OutputSanitizer:
    """
    Sanitize LLM outputs before returning to users.
    
    Removes potentially harmful content and ensures proper formatting.
    """
    
    def sanitize(self, output: str) -> str:
        """
        Sanitize LLM output.
        
        Args:
            output: Raw LLM output
        
        Returns:
            Sanitized output
        """
        # Remove any embedded code execution attempts
        sanitized = re.sub(r'```(?:python|javascript|bash|shell).*?```', 
                          '[CODE_REMOVED]', output, flags=re.DOTALL)
        
        # Remove potential script injections
        sanitized = re.sub(r'<script.*?>.*?</script>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        
        return sanitized
    
    def format_financial_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Format financial data for safe display."""
        formatted = {}
        for key, value in output.items():
            if isinstance(value, float):
                # Format currency values
                if 'amount' in key.lower() or 'revenue' in key.lower():
                    formatted[key] = f"${value:,.2f}"
                else:
                    formatted[key] = round(value, 4)
            else:
                formatted[key] = value
        return formatted


class ContentModerator:
    """
    Moderate content to ensure it meets compliance requirements.
    """
    
    FINANCIAL_DISCLAIMER = (
        "\n\n*Disclaimer: This analysis is for informational purposes only "
        "and does not constitute financial, legal, or investment advice. "
        "Please consult with qualified professionals before making any decisions.*"
    )
    
    def add_disclaimer(self, content: str, content_type: str) -> str:
        """Add appropriate disclaimer based on content type."""
        if content_type in ["financial", "legal", "investment"]:
            return content + self.FINANCIAL_DISCLAIMER
        return content
    
    def check_confidence_level(
        self, 
        confidence: float, 
        min_threshold: float = 0.6
    ) -> tuple[bool, str]:
        """
        Check if confidence level meets minimum threshold.
        
        Returns:
            Tuple of (meets_threshold, warning_message)
        """
        if confidence < min_threshold:
            return False, (
                f"Low confidence ({confidence:.2%}). Results should be "
                "verified by domain experts before use."
            )
        return True, ""
