"""
Intent Classification Module for M&A Due Diligence Platform.

This module provides intent classification to determine whether a user query
should trigger the full M&A due diligence agent chain or be handled directly.

The chain is ONLY activated when:
1. User intent is classified as MA_DUE_DILIGENCE
2. Company names (acquirer and/or target) are explicitly mentioned
"""

from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel, Field
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from src.config.llm_config import get_llm
from src.common.logging_config import get_logger

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
