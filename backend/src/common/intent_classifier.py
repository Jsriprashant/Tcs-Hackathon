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

from enum import Enum
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from src.config.llm_config import get_llm
from src.common.logging_config import get_logger

# Import enhanced models
from src.supervisor.models import (
    AnalysisScope,
    DealType,
    EnhancedIntentResult,
)

logger = get_logger(__name__)


class IntentType(str, Enum):
    """Enum for classifying user query intent."""
    
    MA_DUE_DILIGENCE = "MA_DUE_DILIGENCE"  # Full M&A analysis with company names
    MA_QUESTION = "MA_QUESTION"  # M&A-related question without specific companies
    INFORMATIONAL = "INFORMATIONAL"  # General knowledge/educational query
    GREETING = "GREETING"  # Simple greeting
    HELP = "HELP"  # Asking about capabilities
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
# DOMAIN-SPECIFIC KEYWORDS FOR TARGETED ANALYSIS (NEW)
# =============================================================================

DOMAIN_KEYWORDS = {
    "finance": [
        "financial", "revenue", "profit", "cash flow", "debt", "valuation",
        "balance sheet", "income statement", "ebitda", "margins", "liquidity",
        "solvency", "financial health", "earnings", "assets", "liabilities",
        "profitability", "cash", "fiscal", "accounting", "audit", "budget",
        "cost", "expense", "income", "investment", "capital", "equity",
        "financial risk", "financial analysis", "financial due diligence"
    ],
    "legal": [
        "legal", "litigation", "lawsuit", "contract", "ip", "patent",
        "trademark", "regulatory", "court", "dispute", "sue", "sued",
        "intellectual property", "license", "agreement", "legal risk",
        "attorney", "lawyer", "jurisdiction", "liability", "indemnification",
        "legal due diligence", "legal analysis", "legal review", "contracts"
    ],
    "hr": [
        "hr", "human resources", "employee", "attrition", "retention",
        "workforce", "talent", "culture", "headcount", "key person",
        "compensation", "benefits", "union", "labor", "staff", "hiring",
        "termination", "severance", "pension", "payroll", "hr risk",
        "hr due diligence", "hr analysis", "people", "organizational"
    ],
    "compliance": [
        "compliance", "regulatory", "audit", "sox", "gdpr", "hipaa",
        "environmental", "safety", "osha", "fda", "sec", "violations",
        "non-compliance", "regulation", "policy", "governance", "ethics",
        "compliance risk", "compliance review", "regulatory risk"
    ],
    "strategic": [
        "synergy", "strategic", "market share", "competitive", "growth",
        "integration", "value creation", "positioning", "expansion",
        "strategy", "competitive advantage", "market position", "synergies",
        "strategic fit", "strategic analysis", "strategic review"
    ]
}

# Keywords indicating analysis scope
SCOPE_KEYWORDS = {
    AnalysisScope.FULL_DUE_DILIGENCE: [
        "complete", "comprehensive", "full", "thorough", "detailed",
        "all aspects", "entire", "whole", "everything", "360"
    ],
    AnalysisScope.FINANCIAL_ONLY: [
        "only financial", "just financial", "financial only",
        "financial analysis", "financial due diligence", "financials only"
    ],
    AnalysisScope.LEGAL_ONLY: [
        "only legal", "just legal", "legal only",
        "legal analysis", "legal due diligence", "legal review only"
    ],
    AnalysisScope.HR_ONLY: [
        "only hr", "just hr", "hr only", "human resources only",
        "hr analysis", "hr due diligence", "people only", "workforce only"
    ],
    AnalysisScope.COMPLIANCE_ONLY: [
        "only compliance", "just compliance", "compliance only",
        "compliance analysis", "compliance review", "regulatory only"
    ],
    AnalysisScope.RISK_ASSESSMENT: [
        "risk only", "just risks", "risk assessment", "risk analysis",
        "what are the risks", "identify risks", "risk evaluation"
    ],
    AnalysisScope.COMPARISON: [
        "compare", "comparison", "versus", "vs", "which is better",
        "between", "rank", "best target", "evaluate targets"
    ],
    AnalysisScope.QUICK_OVERVIEW: [
        "quick", "brief", "summary", "overview", "snapshot",
        "high level", "at a glance", "quick look", "fast"
    ]
}

# Deal type keywords
DEAL_TYPE_KEYWORDS = {
    DealType.MERGER: ["merger", "merge", "merging", "consolidation"],
    DealType.ACQUISITION: ["acquisition", "acquire", "acquiring", "takeover", "buyout", "buy"],
    DealType.DIVESTITURE: ["divestiture", "divest", "sell off", "spinoff", "spin-off"],
    DealType.JOINT_VENTURE: ["joint venture", "jv", "partnership", "strategic alliance"],
    DealType.ASSET_PURCHASE: ["asset purchase", "asset deal", "buy assets"]
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

2. **MA_QUESTION** - User has a question about M&A concepts but NO specific company names
   - Examples:
     - "What is due diligence?" → MA_QUESTION
     - "How do synergies work in M&A?" → MA_QUESTION
     - "What are the steps in an acquisition?" → MA_QUESTION
     - "Explain financial due diligence" → MA_QUESTION

3. **INFORMATIONAL** - General question NOT related to M&A
   - Examples:
     - "What's the weather?" → INFORMATIONAL
     - "Tell me a joke" → INFORMATIONAL
     - "What is machine learning?" → INFORMATIONAL

4. **GREETING** - Simple greeting
   - Examples: "Hello", "Hi there", "Good morning"

5. **HELP** - Asking about platform capabilities
   - Examples: "What can you do?", "Help me", "Show me your capabilities"

IMPORTANT RULES:
- For MA_DUE_DILIGENCE, you MUST extract company names from the query
- Generic terms like "company", "target", "firm" without specific names are NOT valid company names
- If the query mentions M&A concepts but no specific company, classify as MA_QUESTION
- Be conservative: when uncertain between MA_DUE_DILIGENCE and MA_QUESTION, choose MA_QUESTION

User Query: {query}

Respond with a JSON object ONLY (no markdown, no explanation outside JSON):
{{
    "intent": "INTENT_TYPE",
    "confidence": 0.0-1.0,
    "acquirer_company": "extracted name or null",
    "target_company": "extracted name or null",
    "reasoning": "brief one-line explanation"
}}"""


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


def classify_intent(query: str) -> IntentClassificationResult:
    """
    Classify user query intent using LLM.
    
    This is the main function to determine whether the agent chain should be activated.
    
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
    
    # Check if query has M&A keywords - if not, likely informational
    if not has_ma_keywords(query):
        # No M&A keywords, but let's still use LLM to be sure
        # It might be a valid M&A request phrased differently
        pass
    
    # Use LLM for classification
    try:
        llm = get_llm(temperature=0.0)  # Deterministic for classification
        
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        messages = [SystemMessage(content=prompt)]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Parse JSON response
        result = parse_llm_response(content, query)
        
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
        # Default to MA_QUESTION if it has M&A keywords, else INFORMATIONAL
        if has_ma_keywords(query):
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


def detect_analysis_scope(query: str) -> AnalysisScope:
    """
    Detect analysis scope from query using keyword matching.
    
    Args:
        query: User query string
        
    Returns:
        AnalysisScope enum value
    """
    query_lower = query.lower()
    
    # Check for specific scope keywords first (more specific matches)
    for scope, keywords in SCOPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                logger.debug(f"Detected scope {scope.value} from keyword: {keyword}")
                return scope
    
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


def detect_deal_type(query: str) -> DealType:
    """
    Detect deal type from query.
    
    Args:
        query: User query string
        
    Returns:
        DealType enum value
    """
    query_lower = query.lower()
    
    for deal_type, keywords in DEAL_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return deal_type
    
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


def classify_intent_enhanced(query: str) -> EnhancedIntentResult:
    """
    Enhanced intent classification with scope detection.
    
    This function provides granular analysis scope detection in addition
    to basic intent classification.
    
    Args:
        query: User query string
        
    Returns:
        EnhancedIntentResult with full classification details
    """
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


def parse_scope(scope_str: Optional[str]) -> Optional[AnalysisScope]:
    """Parse scope string to enum."""
    if not scope_str:
        return None
    try:
        return AnalysisScope(scope_str.upper())
    except ValueError:
        return None


def parse_deal_type(deal_str: Optional[str]) -> Optional[DealType]:
    """Parse deal type string to enum."""
    if not deal_str:
        return None
    try:
        return DealType(deal_str.lower())
    except ValueError:
        return None

