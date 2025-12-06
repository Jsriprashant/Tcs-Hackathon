"""
Intent Classification Module for M&A Due Diligence Platform.

This module provides intent classification to determine whether a user query
should trigger the full M&A due diligence agent chain or be handled directly.

ENHANCED (v2.0):
- Granular analysis scope detection (FULL, FINANCIAL_ONLY, LEGAL_ONLY, etc.)
- Domain keyword detection for targeted analysis
- Deal type identification
- Comparison analysis support

The chain is ONLY activated when:
1. User intent is classified as MA_DUE_DILIGENCE
2. Company names (acquirer and/or target) are explicitly mentioned
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from src.config.llm_config import get_llm
from src.common.logging_config import get_logger

# NOTE: Import enhanced models at function level to avoid circular import
# from src.supervisor.models import AnalysisScope, DealType, EnhancedIntentResult
# These are imported lazily in functions that need them

logger = get_logger(__name__)


def _get_supervisor_models():
    """Lazy import of supervisor models to avoid circular import."""
    from src.supervisor.models import (
        AnalysisScope,
        DealType,
        EnhancedIntentResult,
    )
    return AnalysisScope, DealType, EnhancedIntentResult


class IntentType(str, Enum):
    """Enum for classifying user query intent."""
    
    MA_DUE_DILIGENCE = "MA_DUE_DILIGENCE"  # Full M&A analysis with company names
    MA_QUESTION = "MA_QUESTION"  # M&A-related question without specific companies (conceptual)
    DOMAIN_QUERY_NO_CONTEXT = "DOMAIN_QUERY_NO_CONTEXT"  # Actionable domain query but missing company context
    INFORMATIONAL = "INFORMATIONAL"  # General knowledge/educational query
    GREETING = "GREETING"  # Simple greeting
    HELP = "HELP"  # Asking about capabilities
    FOLLOW_UP = "FOLLOW_UP"  # Follow-up question referencing previous context
    UNKNOWN = "UNKNOWN"  # Unclassifiable


class IntentClassificationResult(BaseModel):
    """Result of intent classification."""
    
    intent: IntentType = Field(description="Classified intent type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    acquirer_company: Optional[str] = Field(default=None, description="Extracted acquirer company name")
    target_company: Optional[str] = Field(default=None, description="Extracted target company name")
    reasoning: str = Field(default="", description="Brief explanation of classification")
    should_activate_chain: bool = Field(default=False, description="Whether to activate the agent chain")


# Greeting patterns for quick detection (avoid LLM call)
GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "whats up", "sup", "yo",
    "hi there", "hello there", "hey there", "hola", "namaste"
]

# Help patterns for quick detection
HELP_PATTERNS = [
    "help", "what can you do", "what do you do", "how can you help",
    "what are your capabilities", "what services", "tell me about yourself",
    "who are you", "what is this", "how does this work", "capabilities"
]

# M&A keywords that suggest (but don't confirm) M&A intent
MA_KEYWORDS = [
    "merger", "acquisition", "acquire", "acquiring", "merge", "merging",
    "due diligence", "takeover", "buyout", "m&a", "deal", "transaction",
    "target company", "acquirer", "bidder", "consolidation"
]

# =============================================================================
# ACTIONABLE QUERY PATTERNS (Queries that need company context)
# =============================================================================

# Patterns indicating an actionable query (not just conceptual)
ACTIONABLE_QUERY_PATTERNS = [
    # "How is X performing" type queries
    r"how\s+(is|are|does)\s+(the\s+)?(company|it|they)",
    r"how\s+is\s+\w+\s+(doing|performing|good)",  # "how is BBD doing with profit"
    r"how\s+\w+\s+is\s+(good|bad|performing)",    # "how BBD is good with profit"
    r"what('s|\s+is)\s+(the\s+)?(company|it|their)",
    r"give\s+me\s+(the\s+)?(breakdown|analysis|details|report|full)",
    r"analyze\s+(the\s+)?(company|it|this|that|\w+\s+policies|\w+\s+health)",
    r"show\s+me\s+(the\s+)?(financials?|revenue|profit|legal|hr)",
    r"tell\s+me\s+about\s+(the\s+)?(company|it|their)",
    
    # Company-specific actionable patterns (with company codes)
    r"\b[A-Z]{2,5}\b.*(profit|revenue|financial|legal|hr|risk)",  # "BBD profit analysis"
    r"(profit|revenue|financial|legal|hr|risk).*\b[A-Z]{2,5}\b",  # "profit of BBD"
    r"\b[A-Z]{2,5}\b\s+(analysis|review|assessment|health)",       # "BBD analysis"
    
    # Imperative queries - must be at start
    r"^(analyze|review|check|assess|evaluate|examine|audit)\b",
    r"^(get|fetch|retrieve|pull|show)\s+(me\s+)?",
    
    # Performance queries
    r"(performing|performance)\s+(on|in|annually|quarterly)",
    r"(annual|quarterly|monthly)\s+(revenue|profit|growth|report)",
    
    # Risk queries - enhanced patterns
    r"(what|identify|find|list)\s+(are\s+)?(the\s+)?(legal\s+)?risks?",
    r"risk\s+(analysis|assessment|evaluation|profile)",
    r"^what\s+are\s+the\s+(legal|financial|hr|compliance)\s+",
    
    # Domain-specific actionable patterns
    r"(full|complete|detailed)\s+breakdown",
    r"(financial|legal|hr)\s+(health|status|analysis|review|risks?)",
    r"policies\s*(analysis|review|comparison)?$",
    
    # Natural language M&A analyst questions
    r"(should\s+we|can\s+we|is\s+it\s+worth)\s+(buy|acquire|merge)",
    r"(due\s+diligence|dd)\s+(on|for)\s+",
    r"(target|acquisition)\s+(analysis|review|assessment)",
    r"what.*(red\s+flags?|concerns?|issues?|problems?)",
    r"(key|main|top)\s+(risks?|concerns?|issues?)",
    
    # Question patterns that imply analysis request
    r"^is\s+\w+\s+(a\s+)?(good|bad|risky|safe)\s+(target|company|acquisition)",
    r"^why\s+(is|are|should)\s+\w+",  # "why is BBD a good target"
]

# Generic references that indicate follow-up or need context
# These are PHRASES not single words - must match as complete phrases
GENERIC_COMPANY_REFERENCES_PHRASES = [
    "the company", "this company", "that company", "the target",
    "the firm", "this firm", "the organization", "this business",
    "the acquisition target", "target company", "the acquirer",
]

# Single word references - must be checked with word boundaries
# and not be part of commands like "show me" or "tell me"
GENERIC_PRONOUN_REFERENCES = [
    # Word boundary patterns to avoid false positives
    r"\bit\b(?!\s+is\s+worth)",  # "it" but not "it is worth" which implies company
    r"\bthey\b",
    r"\btheir\b(?!\s+\w+\s+analysis)",  # "their" but not "their financial analysis"
    r"\bthem\b",
    r"\bits\b",
]

# Conceptual M&A question patterns (educational, not actionable)
CONCEPTUAL_PATTERNS = [
    r"^what\s+is\s+(a\s+)?(merger|acquisition|due\s+diligence|synergy)",
    r"^how\s+(do|does|to)\s+(merger|acquisition|due\s+diligence|synergies?)",
    r"^explain\s+",  # "explain X" is educational
    r"^define\s+",
    r"^what\s+are\s+the\s+(steps|stages|phases|types)\s+(in|of)",
    r"in\s+general",
    r"typically",
    r"usually",
    r"how\s+do\s+\w+\s+work",  # "how do X work" is conceptual
    r"what\s+does\s+\w+\s+mean",  # "what does X mean" is conceptual
    r"^tell\s+me\s+about\s+(the\s+)?(concept|process|meaning)",  # educational
]

# =============================================================================
# DOMAIN-SPECIFIC KEYWORDS FOR TARGETED ANALYSIS (ENHANCED)
# =============================================================================

DOMAIN_KEYWORDS = {
    "finance": [
        # Core financial terms
        "financial", "revenue", "profit", "cash flow", "debt", "valuation",
        "balance sheet", "income statement", "ebitda", "margins", "liquidity",
        "solvency", "financial health", "earnings", "assets", "liabilities",
        "profitability", "cash", "fiscal", "accounting", "audit", "budget",
        "cost", "expense", "income", "investment", "capital", "equity",
        "financial risk", "financial analysis", "financial due diligence",
        # M&A specific financial terms
        "good with profit", "performing on", "doing with revenue",
        "money", "funds", "funding", "sales", "gross", "net income",
        "operating income", "turnover", "quarterly", "annual", "yearly",
        "financial performance", "fiscal year", "fy", "q1", "q2", "q3", "q4",
        "roi", "return on investment", "return on equity", "roe", "roa",
        "working capital", "capex", "capital expenditure", "opex",
        "burn rate", "runway", "gross margin", "net margin", "ebit",
        "enterprise value", "ev", "market cap", "stock price",
        # Natural language patterns
        "making money", "losing money", "profitable", "unprofitable",
        "cash position", "debt load", "leverage ratio", "interest coverage",
        "how much", "what is the value", "worth", "net worth"
    ],
    "legal": [
        # Core legal terms
        "legal", "litigation", "lawsuit", "contract", "ip", "patent",
        "trademark", "regulatory", "court", "dispute", "sue", "sued",
        "intellectual property", "license", "agreement", "legal risk",
        "attorney", "lawyer", "jurisdiction", "liability", "indemnification",
        "legal due diligence", "legal analysis", "legal review", "contracts",
        # M&A specific legal terms
        "pending cases", "active litigation", "legal exposure", "lawsuits",
        "settlements", "class action", "arbitration", "claims", "legal issues",
        "copyright", "trade secret", "nda", "non-compete", "employment agreement",
        "merger agreement", "acquisition agreement", "term sheet",
        "representations", "warranties", "indemnities", "covenants",
        "regulatory approval", "antitrust", "ftc", "doj", "sec filing",
        "material contracts", "change of control", "assignment clause",
        # Natural language patterns
        "any lawsuits", "any legal problems", "being sued", "legal trouble",
        "court cases", "legal matters", "legal standing", "legal situation"
    ],
    "hr": [
        # Core HR terms
        "hr", "human resources", "employee", "attrition", "retention",
        "workforce", "talent", "culture", "headcount", "key person",
        "compensation", "benefits", "union", "labor", "staff", "hiring",
        "termination", "severance", "pension", "payroll", "hr risk",
        "hr due diligence", "hr analysis", "people", "organizational",
        # M&A specific HR terms
        "key employees", "key personnel", "key executives", "management team",
        "employee turnover", "retention rate", "attrition rate", "churn",
        "golden parachute", "retention bonus", "stay bonus", "change in control",
        "employment contracts", "non-compete agreements", "severance packages",
        "organizational structure", "reporting structure", "org chart",
        "salary bands", "pay scales", "equity compensation", "stock options",
        "employee satisfaction", "engagement score", "glassdoor", "morale",
        "diversity", "dei", "inclusion", "hr policies", "employee handbook",
        # Natural language patterns
        "how many employees", "who are the key people", "leadership team",
        "culture like", "people issues", "workforce size", "staff count",
        "employee count", "team size", "who runs", "who leads"
    ],
    "compliance": [
        # Core compliance terms
        "compliance", "regulatory", "audit", "sox", "gdpr", "hipaa",
        "environmental", "safety", "osha", "fda", "sec", "violations",
        "non-compliance", "regulation", "policy", "governance", "ethics",
        "compliance risk", "compliance review", "regulatory risk",
        # M&A specific compliance terms
        "regulatory issues", "compliance issues", "audit findings",
        "internal controls", "control deficiencies", "material weakness",
        "regulatory filings", "licenses", "permits", "certifications",
        "iso", "soc", "soc2", "pci", "data protection", "privacy",
        "whistleblower", "anti-corruption", "fcpa", "uk bribery act",
        "sanctions", "export controls", "trade compliance",
        # Natural language patterns
        "following rules", "breaking rules", "any violations", "in trouble with"
    ],
    "strategic": [
        # Core strategic terms
        "synergy", "strategic", "market share", "competitive", "growth",
        "integration", "value creation", "positioning", "expansion",
        "strategy", "competitive advantage", "market position", "synergies",
        "strategic fit", "strategic analysis", "strategic review",
        # M&A specific strategic terms
        "fit with", "good fit", "good match", "compatible", "complementary",
        "market opportunity", "growth potential", "upside", "value drivers",
        "cost synergies", "revenue synergies", "cross-sell", "upsell",
        "customer overlap", "product overlap", "geographic expansion",
        "strategic rationale", "deal rationale", "why acquire",
        "competitive landscape", "competitors", "market dynamics",
        "industry trends", "market trends", "tailwinds", "headwinds",
        # M&A action terms - these imply strategic decision
        "acquire", "acquisition", "buy", "merge", "merger", "takeover",
        "should we acquire", "should we buy", "should we merge",
        # Natural language patterns
        "good target", "good acquisition", "worth acquiring", "should we buy",
        "makes sense", "strategic value", "why is this company", "potential of"
    ]
}

# Keywords indicating analysis scope (using string keys to avoid circular import)
SCOPE_KEYWORDS = {
    "FULL_DUE_DILIGENCE": [
        "complete", "comprehensive", "full", "thorough", "detailed",
        "all aspects", "entire", "whole", "everything", "360"
    ],
    "FINANCIAL_ONLY": [
        "only financial", "just financial", "financial only",
        "financial analysis", "financial due diligence", "financials only"
    ],
    "LEGAL_ONLY": [
        "only legal", "just legal", "legal only",
        "legal analysis", "legal due diligence", "legal review only"
    ],
    "HR_ONLY": [
        "only hr", "just hr", "hr only", "human resources only",
        "hr analysis", "hr due diligence", "people only", "workforce only"
    ],
    "COMPLIANCE_ONLY": [
        "only compliance", "just compliance", "compliance only",
        "compliance analysis", "compliance review", "regulatory only"
    ],
    "RISK_ASSESSMENT": [
        "risk only", "just risks", "risk assessment", "risk analysis",
        "what are the risks", "identify risks", "risk evaluation"
    ],
    "COMPARISON": [
        "compare", "comparison", "versus", "vs", "which is better",
        "between", "rank", "best target", "evaluate targets"
    ],
    "QUICK_OVERVIEW": [
        "quick", "brief", "summary", "overview", "snapshot",
        "high level", "at a glance", "quick look", "fast"
    ]
}

# Deal type keywords (using string keys to avoid circular import)
DEAL_TYPE_KEYWORDS = {
    "merger": ["merger", "merge", "merging", "consolidation"],
    "acquisition": ["acquisition", "acquire", "acquiring", "takeover", "buyout", "buy"],
    "divestiture": ["divestiture", "divest", "sell off", "spinoff", "spin-off"],
    "joint_venture": ["joint venture", "jv", "partnership", "strategic alliance"],
    "asset_purchase": ["asset purchase", "asset deal", "buy assets"]
}

# Intent classification prompt
INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for an M&A Due Diligence Platform.

Analyze the user query and classify the intent into one of these categories:

1. **MA_DUE_DILIGENCE** - User wants to perform due diligence analysis for specific companies
   - MUST mention at least one company name for merger/acquisition analysis
   - Examples:
     - "Analyze TechCorp for acquisition" → MA_DUE_DILIGENCE (target: TechCorp)
     - "Due diligence on merger between BBD and XYZ" → MA_DUE_DILIGENCE (acquirer: BBD, target: XYZ)
     - "What are the risks of acquiring StartupABC?" → MA_DUE_DILIGENCE (target: StartupABC)
     - "Run financial analysis on Supernova Inc" → MA_DUE_DILIGENCE (target: Supernova Inc)
     - "how is BBD performing on profit" → MA_DUE_DILIGENCE (target: BBD)
     - "tell me about BBD's revenue" → MA_DUE_DILIGENCE (target: BBD)
     - "why is XYZ Corp not profitable" → MA_DUE_DILIGENCE (target: XYZ Corp)

2. **FOLLOW_UP** - User is asking a follow-up question about a previous analysis
   - Uses pronouns like "they", "it", "this company", "the company", "that"
   - References "the same company", "this target", "more about"
   - Asks for additional details about something discussed before
   - Examples:
     - "why is it not profitable?" → FOLLOW_UP (no specific company, referring to previous context)
     - "tell me more about their legal issues" → FOLLOW_UP
     - "what about the HR risks?" → FOLLOW_UP
     - "and the financials?" → FOLLOW_UP
     - "why is that not good?" → FOLLOW_UP
     - "can you also check the legal status?" → FOLLOW_UP

3. **MA_QUESTION** - User has a question about M&A concepts but NO specific company names
   - Examples:
     - "What is due diligence?" → MA_QUESTION
     - "How do synergies work in M&A?" → MA_QUESTION
     - "What are the steps in an acquisition?" → MA_QUESTION
     - "Explain financial due diligence" → MA_QUESTION

4. **INFORMATIONAL** - General question NOT related to M&A
   - Examples:
     - "What's the weather?" → INFORMATIONAL
     - "Tell me a joke" → INFORMATIONAL
     - "What is machine learning?" → INFORMATIONAL

5. **GREETING** - Simple greeting
   - Examples: "Hello", "Hi there", "Good morning"

6. **HELP** - Asking about platform capabilities
   - Examples: "What can you do?", "Help me", "Show me your capabilities"

IMPORTANT RULES:
- For MA_DUE_DILIGENCE, you MUST extract company names from the query
- Company names can be: full names (TechCorp), abbreviations (BBD, TCS), ticker symbols, or codes
- Short 2-4 letter uppercase words in the query are likely company codes/tickers - EXTRACT them as company names
- Generic terms like "company", "target", "firm" without specific names are NOT valid company names
- If the query asks about a SPECIFIC company's financials, legal status, HR, etc. → MA_DUE_DILIGENCE
- If the query asks about M&A concepts in general without a company → MA_QUESTION
- If the query uses pronouns/references to previous context without naming a company → FOLLOW_UP
- Be LIBERAL in extracting company names - if it looks like a company identifier, extract it

User Query: {query}

Respond with a JSON object ONLY (no markdown, no explanation outside JSON):
{{
    "intent": "INTENT_TYPE",
    "confidence": 0.0-1.0,
    "acquirer_company": "extracted name or null",
    "target_company": "extracted name or null",
    "reasoning": "brief one-line explanation"
}}"""


# =============================================================================
# COMPANY NAME DETECTION (NEW)
# =============================================================================

# Common company suffixes and patterns
COMPANY_SUFFIXES = ["inc", "corp", "ltd", "llc", "plc", "co", "company", "group", "holdings"]

def extract_potential_company_names(query: str) -> List[str]:
    """
    Extract potential company names from query using pattern matching.
    
    Looks for:
    1. Capitalized words (2-10 chars) that could be company codes/tickers
    2. Words followed by company suffixes
    3. Known company patterns (acronyms like BBD, TCS, etc.)
    
    Args:
        query: User query string
        
    Returns:
        List of potential company names found
    """
    potential_names = []
    
    # Pattern 1: Look for capitalized 2-5 letter words (likely company codes/tickers)
    # e.g., "BBD", "TCS", "ABC"
    ticker_pattern = r'\b([A-Z]{2,5})\b'
    tickers = re.findall(ticker_pattern, query)
    
    # Filter out common non-company words
    common_words = {"THE", "AND", "FOR", "WITH", "HOW", "WHY", "WHAT", "WHEN", "WHERE", 
                    "WHO", "ARE", "CAN", "HAS", "HAD", "WAS", "NOT", "BUT", "ITS", "OUR"}
    for ticker in tickers:
        if ticker not in common_words:
            potential_names.append(ticker)
    
    # Pattern 2: Words with company suffixes
    for suffix in COMPANY_SUFFIXES:
        pattern = rf'\b(\w+\s*{suffix})\b'
        matches = re.findall(pattern, query, re.IGNORECASE)
        potential_names.extend(matches)
    
    # Pattern 3: Quoted names
    quoted_pattern = r'["\']([^"\']+)["\']'
    quoted = re.findall(quoted_pattern, query)
    potential_names.extend(quoted)
    
    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for name in potential_names:
        name_clean = name.strip()
        if name_clean.lower() not in seen and len(name_clean) >= 2:
            seen.add(name_clean.lower())
            unique_names.append(name_clean)
    
    return unique_names


def quick_intent_check(query: str) -> Optional[IntentClassificationResult]:
    """
    Perform quick pattern-based intent check to avoid LLM call for simple cases.
    
    Args:
        query: User query string
        
    Returns:
        IntentClassificationResult if quick match found, None otherwise
    """
    if not query:
        return IntentClassificationResult(
            intent=IntentType.UNKNOWN,
            confidence=1.0,
            reasoning="Empty query"
        )
    
    query_lower = query.lower().strip()
    
    # Check for greetings (exact match or prefix match)
    for pattern in GREETING_PATTERNS:
        if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.startswith(pattern + "!"):
            return IntentClassificationResult(
                intent=IntentType.GREETING,
                confidence=0.95,
                reasoning=f"Matched greeting pattern: {pattern}"
            )
    
    # Check for help requests
    for pattern in HELP_PATTERNS:
        if pattern in query_lower:
            return IntentClassificationResult(
                intent=IntentType.HELP,
                confidence=0.9,
                reasoning=f"Matched help pattern: {pattern}"
            )
    
    # Check for very short queries that are likely greetings
    words = query_lower.split()
    if len(words) <= 2 and any(g in query_lower for g in ["hi", "hello", "hey"]):
        return IntentClassificationResult(
            intent=IntentType.GREETING,
            confidence=0.85,
            reasoning="Short greeting-like query"
        )
    
    return None  # Need LLM for classification


def has_ma_keywords(query: str) -> bool:
    """
    Check if query contains M&A-related keywords.
    
    Args:
        query: User query string
        
    Returns:
        True if M&A keywords found
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in MA_KEYWORDS)


def is_actionable_query(query: str) -> bool:
    """
    Check if query is an actionable request (vs conceptual/educational).
    
    Actionable queries are requests for specific analysis or data,
    not general questions about concepts.
    
    Args:
        query: User query string
        
    Returns:
        True if query appears actionable
    """
    query_lower = query.lower()
    
    # Check for actionable patterns
    for pattern in ACTIONABLE_QUERY_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    
    return False


def is_conceptual_query(query: str) -> bool:
    """
    Check if query is a conceptual/educational question.
    
    Args:
        query: User query string
        
    Returns:
        True if query is conceptual
    """
    query_lower = query.lower()
    
    for pattern in CONCEPTUAL_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    
    return False


def has_generic_company_reference(query: str) -> bool:
    """
    Check if query references a company generically (without specific name).
    
    ENHANCED: Uses word boundaries and excludes false positives like
    "show me" or "tell me" where "me" is not a company reference.
    
    Args:
        query: User query string
        
    Returns:
        True if generic company reference found
    """
    query_lower = query.lower()
    
    # First check exact phrase matches
    for ref in GENERIC_COMPANY_REFERENCES_PHRASES:
        if ref in query_lower:
            return True
    
    # Check pronoun patterns with word boundaries
    for pattern in GENERIC_PRONOUN_REFERENCES:
        if re.search(pattern, query_lower):
            # Additional check: if query has a specific company code, skip generic ref
            # E.g., "show me BBD profit" - "me" is not a company reference
            potential_companies = extract_potential_company_names(query)
            if potential_companies:
                # Has a specific company - don't consider this a generic ref
                return False
            return True
    
    return False


def has_domain_keywords(query: str) -> Tuple[bool, List[str]]:
    """
    Check if query contains domain-specific keywords.
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (has_keywords, list of matched domains)
    """
    query_lower = query.lower()
    matched_domains = []
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                if domain not in matched_domains:
                    matched_domains.append(domain)
                break
    
    return (len(matched_domains) > 0, matched_domains)


def classify_intent(query: str) -> IntentClassificationResult:
    """
    Classify user query intent using LLM with smart fallback detection.
    
    This is the main function to determine whether the agent chain should be activated.
    
    ENHANCED: Now detects actionable domain queries that need clarification
    (e.g., "how is the company performing on revenue" without a company name).
    
    Args:
        query: User query string
        
    Returns:
        IntentClassificationResult with intent type and extracted companies
    """
    # Try quick pattern matching first
    quick_result = quick_intent_check(query)
    if quick_result:
        logger.info(f"Quick intent classification: {quick_result.intent.value}")
        return quick_result
    
    # Pre-check: Detect actionable queries with generic company references
    # These need clarification, not a generic answer
    has_domains, matched_domains = has_domain_keywords(query)
    is_actionable = is_actionable_query(query)
    has_generic_ref = has_generic_company_reference(query)
    is_conceptual = is_conceptual_query(query)
    
    logger.debug(f"Query analysis - domains: {matched_domains}, actionable: {is_actionable}, "
                 f"generic_ref: {has_generic_ref}, conceptual: {is_conceptual}")
    
    # CASE 1: Actionable query with generic company reference → needs clarification
    if is_actionable and has_domains and has_generic_ref and not is_conceptual:
        logger.info(f"Detected actionable domain query without specific company: {matched_domains}")
        return IntentClassificationResult(
            intent=IntentType.DOMAIN_QUERY_NO_CONTEXT,
            confidence=0.85,
            reasoning=f"Actionable {', '.join(matched_domains)} query but no specific company mentioned. "
                     f"User referenced '{_find_generic_ref(query)}' which needs clarification."
        )
    
    # NOTE: We don't early-return for actionable+domain queries without generic ref
    # because they might contain a specific company name that LLM can extract
    # e.g., "analyze the financial health of BBD" has "BBD" as a company name
    
    # Use LLM for classification
    try:
        llm = get_llm(temperature=0.0)  # Deterministic for classification
        
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        messages = [SystemMessage(content=prompt)]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Parse JSON response
        result = parse_llm_response(content, query)
        
        # NEW: Try to extract company names using pattern matching if LLM missed them
        potential_companies = extract_potential_company_names(query)
        logger.debug(f"Potential company names extracted: {potential_companies}")
        
        # If LLM didn't extract companies but we found potential ones, use them
        if result.target_company is None and potential_companies:
            # Use the first potential company as target
            result.target_company = potential_companies[0]
            logger.info(f"Pattern-matched company name: {result.target_company}")
            
            # If query has domain keywords and we now have a company, upgrade to MA_DUE_DILIGENCE
            if has_domains and result.intent in [IntentType.MA_QUESTION, IntentType.INFORMATIONAL, IntentType.DOMAIN_QUERY_NO_CONTEXT]:
                result.intent = IntentType.MA_DUE_DILIGENCE
                result.reasoning = f"Detected company '{result.target_company}' with {', '.join(matched_domains)} focus"
                logger.info(f"Upgraded to MA_DUE_DILIGENCE with target: {result.target_company}")
        
        # Post-process: If LLM says MA_QUESTION or INFORMATIONAL but query is ACTIONABLE
        # with domain keywords and no company was extracted, reclassify as DOMAIN_QUERY_NO_CONTEXT
        # Key: Must be actionable AND not conceptual to reclassify
        if result.intent in [IntentType.MA_QUESTION, IntentType.INFORMATIONAL]:
            if result.acquirer_company is None and result.target_company is None:
                # Only reclassify if query is actionable AND has domains AND is not conceptual
                if is_actionable and has_domains and not is_conceptual:
                    logger.info(f"Reclassifying {result.intent.value} to DOMAIN_QUERY_NO_CONTEXT (actionable+domains: {matched_domains})")
                    result.intent = IntentType.DOMAIN_QUERY_NO_CONTEXT
                    result.reasoning = f"Actionable {', '.join(matched_domains)} query needs company context"
        
        # Set should_activate_chain flag
        result.should_activate_chain = (
            result.intent == IntentType.MA_DUE_DILIGENCE and
            (result.acquirer_company is not None or result.target_company is not None)
        )
        
        logger.info(
            f"LLM intent classification: {result.intent.value}, "
            f"chain_activated: {result.should_activate_chain}, "
            f"acquirer: {result.acquirer_company}, target: {result.target_company}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        
        # Smart fallback based on query characteristics
        if is_actionable and has_domains:
            return IntentClassificationResult(
                intent=IntentType.DOMAIN_QUERY_NO_CONTEXT,
                confidence=0.6,
                reasoning=f"Fallback: actionable {', '.join(matched_domains)} query without company context"
            )
        elif has_ma_keywords(query):
            return IntentClassificationResult(
                intent=IntentType.MA_QUESTION,
                confidence=0.5,
                reasoning=f"LLM classification failed, defaulting based on M&A keywords: {str(e)}"
            )
        return IntentClassificationResult(
            intent=IntentType.INFORMATIONAL,
            confidence=0.5,
            reasoning=f"LLM classification failed: {str(e)}"
        )


def _find_generic_ref(query: str) -> str:
    """Find which generic reference was used in the query."""
    query_lower = query.lower()
    for ref in GENERIC_COMPANY_REFERENCES_PHRASES:
        if ref in query_lower:
            return ref
    return "unknown reference"


def parse_llm_response(content: str, original_query: str) -> IntentClassificationResult:
    """
    Parse LLM JSON response into IntentClassificationResult.
    
    Args:
        content: LLM response content
        original_query: Original user query for fallback
        
    Returns:
        IntentClassificationResult
    """
    try:
        # Try to extract JSON from response
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        elif "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
        else:
            json_str = content
        
        data = json.loads(json_str)
        
        # Map intent string to enum
        intent_str = data.get("intent", "UNKNOWN").upper()
        try:
            intent = IntentType(intent_str)
        except ValueError:
            intent = IntentType.UNKNOWN
        
        return IntentClassificationResult(
            intent=intent,
            confidence=float(data.get("confidence", 0.7)),
            acquirer_company=data.get("acquirer_company") if data.get("acquirer_company") != "null" else None,
            target_company=data.get("target_company") if data.get("target_company") != "null" else None,
            reasoning=data.get("reasoning", "")
        )
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse LLM response: {e}, content: {content[:200]}")
        
        # Fallback: try to infer from response text
        return infer_intent_from_text(content, original_query)


def infer_intent_from_text(content: str, original_query: str) -> IntentClassificationResult:
    """
    Infer intent from LLM response text when JSON parsing fails.
    
    Args:
        content: LLM response content
        original_query: Original user query
        
    Returns:
        IntentClassificationResult with best effort classification
    """
    content_lower = content.lower()
    
    if "ma_due_diligence" in content_lower:
        # Try to extract company names from the response
        acquirer, target = extract_companies_from_text(content)
        return IntentClassificationResult(
            intent=IntentType.MA_DUE_DILIGENCE,
            confidence=0.6,
            acquirer_company=acquirer,
            target_company=target,
            reasoning="Inferred from LLM response text"
        )
    elif "ma_question" in content_lower:
        return IntentClassificationResult(
            intent=IntentType.MA_QUESTION,
            confidence=0.6,
            reasoning="Inferred from LLM response text"
        )
    elif "greeting" in content_lower:
        return IntentClassificationResult(
            intent=IntentType.GREETING,
            confidence=0.6,
            reasoning="Inferred from LLM response text"
        )
    elif "help" in content_lower:
        return IntentClassificationResult(
            intent=IntentType.HELP,
            confidence=0.6,
            reasoning="Inferred from LLM response text"
        )
    else:
        # Default based on original query
        if has_ma_keywords(original_query):
            return IntentClassificationResult(
                intent=IntentType.MA_QUESTION,
                confidence=0.5,
                reasoning="Fallback: query has M&A keywords but no company names"
            )
        return IntentClassificationResult(
            intent=IntentType.INFORMATIONAL,
            confidence=0.5,
            reasoning="Fallback: could not classify"
        )


def extract_companies_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to extract company names from text.
    
    Args:
        text: Text containing potential company names
        
    Returns:
        Tuple of (acquirer_company, target_company)
    """
    acquirer = None
    target = None
    
    # Look for acquirer patterns
    acquirer_patterns = [
        r'acquirer["\s:]+([A-Z][A-Za-z0-9\s&]+?)(?:[",\n]|$)',
        r'acquirer_company["\s:]+([A-Z][A-Za-z0-9\s&]+?)(?:[",\n]|$)',
    ]
    
    for pattern in acquirer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).strip().lower() not in ["null", "none", ""]:
            acquirer = match.group(1).strip()
            break
    
    # Look for target patterns
    target_patterns = [
        r'target["\s:]+([A-Z][A-Za-z0-9\s&]+?)(?:[",\n]|$)',
        r'target_company["\s:]+([A-Z][A-Za-z0-9\s&]+?)(?:[",\n]|$)',
    ]
    
    for pattern in target_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).strip().lower() not in ["null", "none", ""]:
            target = match.group(1).strip()
            break
    
    return acquirer, target


def get_last_human_message(messages: list) -> Optional[str]:
    """
    Extract the last human message content from a list of messages.
    
    Args:
        messages: List of message objects
        
    Returns:
        String content of last human message, or None
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            # Handle multimodal content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return " ".join(text_parts)
            return content
        elif hasattr(msg, 'type') and msg.type == 'human':
            return msg.content if hasattr(msg, 'content') else str(msg)
    
    return None


# =============================================================================
# ENHANCED INTENT CLASSIFICATION (v2.0)
# =============================================================================

ENHANCED_INTENT_PROMPT = """You are an advanced intent classifier for an M&A Due Diligence Platform.

Analyze the user query and extract:
1. Basic intent type
2. Analysis scope (what type of analysis is requested)
3. Company names (acquirer and target)
4. Deal type (merger, acquisition, etc.)
5. Required analysis domains

## Intent Types
- **MA_DUE_DILIGENCE**: User wants analysis for specific companies
- **MA_QUESTION**: M&A question without specific companies
- **INFORMATIONAL**: General non-M&A question
- **GREETING**: Simple greeting
- **HELP**: Platform capabilities question

## Analysis Scope (for MA_DUE_DILIGENCE only)
- **FULL_DUE_DILIGENCE**: Complete analysis (finance + legal + HR + compliance)
- **FINANCIAL_ONLY**: Only financial analysis
- **LEGAL_ONLY**: Only legal analysis  
- **HR_ONLY**: Only HR/people analysis
- **COMPLIANCE_ONLY**: Only compliance/regulatory analysis
- **STRATEGIC_ONLY**: Only strategic/synergy analysis
- **RISK_ASSESSMENT**: Focus on risk identification
- **COMPARISON**: Compare multiple targets
- **QUICK_OVERVIEW**: Brief high-level summary

## Deal Types
- acquisition, merger, divestiture, joint_venture, asset_purchase

## Examples

Query: "Complete due diligence on TechCorp acquisition by MegaCorp"
→ intent: MA_DUE_DILIGENCE, scope: FULL_DUE_DILIGENCE, acquirer: MegaCorp, target: TechCorp

Query: "What are the financial risks of acquiring StartupXYZ?"
→ intent: MA_DUE_DILIGENCE, scope: FINANCIAL_ONLY, target: StartupXYZ

Query: "Legal review for BBD merger"
→ intent: MA_DUE_DILIGENCE, scope: LEGAL_ONLY, target: BBD

Query: "Compare TargetA and TargetB for acquisition"
→ intent: MA_DUE_DILIGENCE, scope: COMPARISON, targets: [TargetA, TargetB]

Query: "Quick overview of AlphaCorp"
→ intent: MA_DUE_DILIGENCE, scope: QUICK_OVERVIEW, target: AlphaCorp

Query: "What is synergy in M&A?"
→ intent: MA_QUESTION

User Query: {query}

Respond with JSON only:
{{
    "intent": "INTENT_TYPE",
    "confidence": 0.0-1.0,
    "analysis_scope": "SCOPE_TYPE",
    "acquirer_company": "name or null",
    "target_company": "name or null",
    "additional_companies": ["list of other companies if comparison"],
    "deal_type": "acquisition|merger|divestiture|joint_venture|asset_purchase",
    "required_domains": ["finance", "legal", "hr", "compliance", "strategic"],
    "priority_domain": "primary focus domain or null",
    "depth": "quick|standard|deep",
    "reasoning": "brief explanation"
}}"""


def detect_analysis_scope(query: str):
    """
    Detect analysis scope from query using keyword matching.
    
    Args:
        query: User query string
        
    Returns:
        AnalysisScope enum value
    """
    AnalysisScope, _, _ = _get_supervisor_models()
    query_lower = query.lower()
    
    # Check for specific scope keywords first (more specific matches)
    for scope_str, keywords in SCOPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                logger.debug(f"Detected scope {scope_str} from keyword: {keyword}")
                return AnalysisScope(scope_str)
    
    # Check domain-specific keywords to infer scope
    domain_matches = detect_required_domains(query)
    
    if len(domain_matches) == 1:
        domain = domain_matches[0]
        scope_map = {
            "finance": AnalysisScope.FINANCIAL_ONLY,
            "legal": AnalysisScope.LEGAL_ONLY,
            "hr": AnalysisScope.HR_ONLY,
            "compliance": AnalysisScope.COMPLIANCE_ONLY,
            "strategic": AnalysisScope.STRATEGIC_ONLY,
        }
        return scope_map.get(domain, AnalysisScope.FULL_DUE_DILIGENCE)
    
    # Default to full due diligence
    return AnalysisScope.FULL_DUE_DILIGENCE


def detect_required_domains(query: str) -> List[str]:
    """
    Detect which analysis domains are required based on query keywords.
    
    Args:
        query: User query string
        
    Returns:
        List of required domain names
    """
    query_lower = query.lower()
    matched_domains = []
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                if domain not in matched_domains:
                    matched_domains.append(domain)
                break
    
    # If no specific domains detected, return all for full DD
    if not matched_domains:
        return ["finance", "legal", "hr", "compliance"]
    
    return matched_domains


def detect_deal_type(query: str):
    """
    Detect deal type from query.
    
    Args:
        query: User query string
        
    Returns:
        DealType enum value
    """
    _, DealType, _ = _get_supervisor_models()
    query_lower = query.lower()
    
    for deal_type_str, keywords in DEAL_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return DealType(deal_type_str)
    
    # Default to acquisition
    return DealType.ACQUISITION


def detect_depth(query: str) -> str:
    """
    Detect analysis depth from query.
    
    Args:
        query: User query string
        
    Returns:
        Depth level: quick, standard, or deep
    """
    query_lower = query.lower()
    
    quick_words = ["quick", "brief", "fast", "summary", "overview", "snapshot"]
    deep_words = ["comprehensive", "thorough", "detailed", "complete", "deep", "full"]
    
    for word in quick_words:
        if word in query_lower:
            return "quick"
    
    for word in deep_words:
        if word in query_lower:
            return "deep"
    
    return "standard"


def classify_intent_enhanced(query: str):
    """
    Enhanced intent classification with scope detection.
    
    This function provides granular analysis scope detection in addition
    to basic intent classification.
    
    Args:
        query: User query string
        
    Returns:
        EnhancedIntentResult with full classification details
    """
    AnalysisScope, DealType, EnhancedIntentResult = _get_supervisor_models()
    
    # First get basic classification
    basic_result = classify_intent(query)
    
    # If not MA_DUE_DILIGENCE, return with defaults
    if basic_result.intent != IntentType.MA_DUE_DILIGENCE:
        return EnhancedIntentResult(
            intent=basic_result.intent.value,
            confidence=basic_result.confidence,
            acquirer_company=basic_result.acquirer_company,
            target_company=basic_result.target_company,
            analysis_scope=AnalysisScope.FULL_DUE_DILIGENCE,
            required_domains=[],
            deal_type=DealType.ACQUISITION,
            depth="standard",
            should_activate_chain=False,
            reasoning=basic_result.reasoning
        )
    
    # Detect analysis scope
    analysis_scope = detect_analysis_scope(query)
    
    # Detect required domains
    required_domains = detect_required_domains(query)
    
    # Detect deal type
    deal_type = detect_deal_type(query)
    
    # Detect depth
    depth = detect_depth(query)
    
    # Determine priority domain (the most emphasized one)
    priority_domain = required_domains[0] if len(required_domains) == 1 else None
    
    # Try LLM-based enhanced classification for better accuracy
    try:
        llm = get_llm(temperature=0.0)
        prompt = ENHANCED_INTENT_PROMPT.format(query=query)
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        
        # Parse enhanced response
        enhanced_data = parse_enhanced_response(content)
        if enhanced_data:
            return EnhancedIntentResult(
                intent=basic_result.intent.value,
                confidence=enhanced_data.get("confidence", basic_result.confidence),
                acquirer_company=enhanced_data.get("acquirer_company") or basic_result.acquirer_company,
                target_company=enhanced_data.get("target_company") or basic_result.target_company,
                additional_companies=enhanced_data.get("additional_companies", []),
                analysis_scope=parse_scope(enhanced_data.get("analysis_scope")) or analysis_scope,
                required_domains=enhanced_data.get("required_domains", required_domains),
                deal_type=parse_deal_type(enhanced_data.get("deal_type")) or deal_type,
                priority_domain=enhanced_data.get("priority_domain") or priority_domain,
                depth=enhanced_data.get("depth", depth),
                should_activate_chain=True,
                reasoning=enhanced_data.get("reasoning", basic_result.reasoning)
            )
    except Exception as e:
        logger.warning(f"Enhanced classification failed, using fallback: {e}")
    
    # Fallback to keyword-based detection
    return EnhancedIntentResult(
        intent=basic_result.intent.value,
        confidence=basic_result.confidence,
        acquirer_company=basic_result.acquirer_company,
        target_company=basic_result.target_company,
        analysis_scope=analysis_scope,
        required_domains=required_domains,
        deal_type=deal_type,
        priority_domain=priority_domain,
        depth=depth,
        should_activate_chain=basic_result.should_activate_chain,
        reasoning=basic_result.reasoning
    )


def parse_enhanced_response(content: str) -> Optional[dict]:
    """
    Parse enhanced LLM response JSON.
    
    Args:
        content: LLM response content
        
    Returns:
        Parsed dictionary or None
    """
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        elif "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
        else:
            json_str = content
        
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Failed to parse enhanced response: {e}")
        return None


def parse_scope(scope_str: Optional[str]):
    """Parse scope string to enum."""
    if not scope_str:
        return None
    try:
        AnalysisScope, _, _ = _get_supervisor_models()
        return AnalysisScope(scope_str.upper())
    except ValueError:
        return None


def parse_deal_type(deal_str: Optional[str]):
    """Parse deal type string to enum."""
    if not deal_str:
        return None
    try:
        _, DealType, _ = _get_supervisor_models()
        return DealType(deal_str.lower())
    except ValueError:
        return None

