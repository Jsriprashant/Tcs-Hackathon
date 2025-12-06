"""
Social Media Opinion Analysis Tools for MCP Server.

Provides tools to:
- Analyze sentiment of user opinions on companies
- Get opinions for a specific company
- Auto-update opinions using LLM every 30 seconds
"""

import json
import os
import random
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except ImportError:
    httpx = None

# Data file path
DATA_DIR = Path(__file__).parent.parent.parent / "data"
OPINIONS_FILE = DATA_DIR / "social_media_opinions.json"

# Sample usernames for generating new opinions
SAMPLE_USERNAMES = [
    "tech_guru_2025", "software_dev_pro", "cloud_architect", "ai_enthusiast",
    "car_fanatic", "ev_advocate", "auto_reviewer", "speed_demon_99",
    "gadget_master", "iot_expert", "smart_home_lover", "tech_critic",
    "green_energy_fan", "solar_advocate", "eco_investor", "sustainability_now",
    "health_conscious", "fitness_tracker", "wellness_guru", "med_tech_user",
    "random_user_123", "daily_reviewer", "honest_opinion", "critical_thinker",
    "happy_customer", "disappointed_buyer", "neutral_observer", "industry_watcher"
]

PLATFORMS = ["Twitter", "LinkedIn", "Reddit", "Facebook", "Instagram", "YouTube"]

# Company-specific topics for generating relevant opinions
COMPANY_TOPICS = {
    "BBD Softwares": [
        "cloud migration", "API services", "enterprise solutions", "developer tools",
        "customer support", "software updates", "security features", "pricing plans",
        "documentation", "integration capabilities", "AI features", "performance"
    ],
    "Supernova": [
        "electric vehicles", "battery range", "autopilot", "charging network",
        "build quality", "customer service", "software updates", "safety features",
        "acceleration", "interior design", "price value", "sustainability"
    ],
    "Technobox": [
        "smart home devices", "IoT integration", "privacy concerns", "product quality",
        "customer support", "app experience", "device compatibility", "pricing",
        "smartwatch features", "speaker quality", "setup process", "durability"
    ],
    "GreenLeaf Energy": [
        "solar panels", "installation service", "energy savings", "customer support",
        "panel efficiency", "warranty", "pricing", "maintenance", "coverage area",
        "environmental impact", "ROI", "battery storage"
    ],
    "HealthFirst": [
        "telemedicine", "app usability", "doctor availability", "prescription service",
        "health monitoring", "fitness integration", "data privacy", "customer support",
        "pricing plans", "specialist access", "wait times", "user interface"
    ]
}


def load_opinions_data() -> dict:
    """Load opinions data from JSON file."""
    if OPINIONS_FILE.exists():
        with open(OPINIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"companies": {}, "opinions": [], "metadata": {}}


def save_opinions_data(data: dict) -> None:
    """Save opinions data to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OPINIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def analyze_sentiment_basic(text: str) -> dict:
    """
    Basic sentiment analysis using keyword matching.
    Returns sentiment score and classification.
    """
    positive_words = [
        "amazing", "excellent", "great", "love", "best", "fantastic", "awesome",
        "impressed", "outstanding", "wonderful", "perfect", "brilliant", "superb",
        "recommend", "satisfied", "happy", "incredible", "innovative", "efficient",
        "seamless", "reliable", "saved", "easy", "smooth", "quality", "valuable"
    ]
    
    negative_words = [
        "disappointed", "terrible", "worst", "hate", "awful", "horrible", "poor",
        "frustrated", "annoying", "useless", "waste", "bad", "fail", "broken",
        "slow", "expensive", "complicated", "difficult", "misleading", "concern",
        "issue", "problem", "bug", "error", "refuse", "nightmare", "joke"
    ]
    
    neutral_words = [
        "okay", "average", "decent", "fine", "acceptable", "moderate", "fair"
    ]
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    neutral_count = sum(1 for word in neutral_words if word in text_lower)
    
    total = positive_count + negative_count + neutral_count
    if total == 0:
        total = 1
    
    # Calculate sentiment score (-1 to 1)
    score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
    
    # Classify sentiment
    if score > 0.2:
        sentiment = "positive"
    elif score < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "score": round(score, 3),
        "sentiment": sentiment,
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "neutral_indicators": neutral_count
    }


def get_company_opinions(company_name: str) -> list[dict]:
    """Get all opinions for a specific company."""
    data = load_opinions_data()
    company_lower = company_name.lower()
    
    matching_opinions = []
    for opinion in data.get("opinions", []):
        if opinion.get("company", "").lower() == company_lower or \
           company_lower in opinion.get("company", "").lower():
            matching_opinions.append(opinion)
    
    return matching_opinions


def analyze_company_sentiment(company_name: str) -> dict[str, Any]:
    """
    Analyze sentiment for all opinions about a specific company.
    
    Args:
        company_name: Name of the company to analyze
        
    Returns:
        Comprehensive sentiment analysis results
    """
    data = load_opinions_data()
    
    # Find matching company
    company_key = None
    for key in data.get("companies", {}):
        if company_name.lower() in key.lower():
            company_key = key
            break
    
    if not company_key:
        available = list(data.get("companies", {}).keys())
        return {
            "success": False,
            "error": f"Company '{company_name}' not found",
            "available_companies": available
        }
    
    # Get company info
    company_info = data["companies"][company_key]
    
    # Get all opinions for this company
    opinions = get_company_opinions(company_key)
    
    if not opinions:
        return {
            "success": True,
            "company": company_key,
            "company_info": company_info,
            "total_opinions": 0,
            "message": "No opinions found for this company"
        }
    
    # Analyze each opinion
    sentiment_results = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_score = 0
    
    for opinion in opinions:
        sentiment = analyze_sentiment_basic(opinion["opinion"])
        sentiment_results.append({
            "user": opinion["user"],
            "opinion": opinion["opinion"],
            "platform": opinion.get("platform", "Unknown"),
            "timestamp": opinion.get("timestamp", ""),
            "likes": opinion.get("likes", 0),
            "sentiment": sentiment
        })
        
        total_score += sentiment["score"]
        if sentiment["sentiment"] == "positive":
            positive_count += 1
        elif sentiment["sentiment"] == "negative":
            negative_count += 1
        else:
            neutral_count += 1
    
    # Calculate overall metrics
    total_opinions = len(opinions)
    avg_score = total_score / total_opinions if total_opinions > 0 else 0
    
    # Determine overall sentiment
    if avg_score > 0.2:
        overall_sentiment = "positive"
    elif avg_score < -0.2:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "mixed"
    
    # Get top positive and negative opinions
    sorted_by_sentiment = sorted(sentiment_results, key=lambda x: x["sentiment"]["score"], reverse=True)
    top_positive = sorted_by_sentiment[:3] if len(sorted_by_sentiment) >= 3 else sorted_by_sentiment
    top_negative = sorted_by_sentiment[-3:] if len(sorted_by_sentiment) >= 3 else []
    
    return {
        "success": True,
        "company": company_key,
        "company_info": company_info,
        "total_opinions": total_opinions,
        "overall_sentiment": overall_sentiment,
        "average_sentiment_score": round(avg_score, 3),
        "sentiment_breakdown": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "positive_percentage": round(positive_count / total_opinions * 100, 1),
            "negative_percentage": round(negative_count / total_opinions * 100, 1),
            "neutral_percentage": round(neutral_count / total_opinions * 100, 1)
        },
        "top_positive_opinions": top_positive,
        "top_negative_opinions": top_negative,
        "all_opinions_analyzed": sentiment_results,
        "analysis_timestamp": datetime.now().isoformat()
    }


async def generate_opinion_with_llm(company_name: str, company_info: dict) -> Optional[dict]:
    """
    Generate a new opinion using LLM API call.
    Falls back to template-based generation if LLM is unavailable.
    """
    topics = COMPANY_TOPICS.get(company_name, ["products", "services", "quality"])
    topic = random.choice(topics)
    
    # Try LLM API call first if httpx is available
    if httpx:
        try:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
            endpoint = os.environ.get("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
            model = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
            
            if api_key:
                prompt = f"""Generate a realistic social media opinion about {company_name}, a {company_info.get('industry', 'company')} company.
Topic focus: {topic}
The opinion should be 1-2 sentences, natural sounding, and can be positive, negative, or neutral.
Just return the opinion text, nothing else."""

                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 100,
                            "temperature": 0.8
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        opinion_text = result["choices"][0]["message"]["content"].strip()
                        return create_opinion_object(company_name, opinion_text)
        except Exception as e:
            print(f"LLM call failed, using template: {e}")
    
    # Fallback to template-based generation
    return generate_template_opinion(company_name, company_info, topic)


def generate_template_opinion(company_name: str, company_info: dict, topic: str) -> dict:
    """Generate opinion using templates when LLM is unavailable."""
    
    positive_templates = [
        f"Really impressed with {company_name}'s {topic}. Exceeded my expectations!",
        f"{company_name} is doing amazing things with their {topic}. Highly recommend!",
        f"Just tried {company_name}'s {topic} and I'm blown away. Great job!",
        f"The {topic} from {company_name} is top-notch. Best in the industry!",
        f"Love how {company_name} handles {topic}. Customer for life!",
    ]
    
    negative_templates = [
        f"Disappointed with {company_name}'s {topic}. Expected much better.",
        f"{company_name} needs to improve their {topic}. Not satisfied at all.",
        f"Had issues with {company_name}'s {topic}. Won't be using again.",
        f"The {topic} from {company_name} is overrated. Save your money.",
        f"Frustrated with {company_name}'s approach to {topic}. Very disappointing.",
    ]
    
    neutral_templates = [
        f"{company_name}'s {topic} is okay. Nothing special but gets the job done.",
        f"Mixed feelings about {company_name}'s {topic}. Has pros and cons.",
        f"Tried {company_name}'s {topic}. It's decent but could be better.",
        f"{company_name}'s {topic} is average compared to competitors.",
        f"Not sure what to think about {company_name}'s {topic}. Time will tell.",
    ]
    
    # Randomly select sentiment type (weighted: 40% positive, 30% negative, 30% neutral)
    sentiment_type = random.choices(
        ["positive", "negative", "neutral"],
        weights=[0.4, 0.3, 0.3]
    )[0]
    
    if sentiment_type == "positive":
        opinion_text = random.choice(positive_templates)
    elif sentiment_type == "negative":
        opinion_text = random.choice(negative_templates)
    else:
        opinion_text = random.choice(neutral_templates)
    
    return create_opinion_object(company_name, opinion_text)


def create_opinion_object(company_name: str, opinion_text: str) -> dict:
    """Create a complete opinion object."""
    data = load_opinions_data()
    opinion_count = len(data.get("opinions", [])) + 1
    
    return {
        "id": f"op_{opinion_count:03d}",
        "user": random.choice(SAMPLE_USERNAMES),
        "company": company_name,
        "opinion": opinion_text,
        "timestamp": datetime.now().isoformat() + "Z",
        "likes": random.randint(10, 500),
        "platform": random.choice(PLATFORMS)
    }


async def update_opinions_periodically():
    """Background task to update opinions every 30 seconds."""
    while True:
        try:
            data = load_opinions_data()
            companies = list(data.get("companies", {}).keys())
            
            if companies:
                # Select a random company to add opinion for
                company_name = random.choice(companies)
                company_info = data["companies"][company_name]
                
                # Generate new opinion
                new_opinion = await generate_opinion_with_llm(company_name, company_info)
                
                if new_opinion:
                    data["opinions"].append(new_opinion)
                    data["metadata"]["last_updated"] = datetime.now().isoformat() + "Z"
                    data["metadata"]["total_opinions"] = len(data["opinions"])
                    save_opinions_data(data)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Added new opinion for {company_name}")
            
        except Exception as e:
            print(f"Error updating opinions: {e}")
        
        await asyncio.sleep(30)  # Wait 30 seconds


def start_background_updater():
    """Start the background opinion updater in a separate thread."""
    def run_async_updater():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_opinions_periodically())
    
    thread = threading.Thread(target=run_async_updater, daemon=True)
    thread.start()
    print("Background opinion updater started (updates every 30 seconds)")
    return thread


# Export functions for use in server.py
__all__ = [
    "analyze_company_sentiment",
    "get_company_opinions", 
    "load_opinions_data",
    "save_opinions_data",
    "start_background_updater"
]