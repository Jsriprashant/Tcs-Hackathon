#!/usr/bin/env python3
"""
Test script to diagnose TCS GenAI Lab API connectivity.

Run from the backend directory:
    python scripts/test_api.py
"""

import os
import sys
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = os.getenv("TCS_GENAI_BASE_URL", "https://genailab.tcs.in")
API_KEY = os.getenv("TCS_GENAI_API_KEY", "")
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() != "false"

# Models to test
MODELS_TO_TEST = [
    os.getenv("TCS_GENAI_MODEL_GPT4O", "azure/genailab-maas-gpt-4o"),
    os.getenv("TCS_GENAI_MODEL_GPT35", "azure/genailab-maas-gpt-35-turbo"),
    os.getenv("TCS_GENAI_MODEL_GPT4O_MINI", "azure/genailab-maas-gpt-4o-mini"),
]


def test_api_connectivity():
    """Test basic API connectivity."""
    print("=" * 60)
    print("TCS GenAI Lab API Connectivity Test")
    print("=" * 60)
    
    print(f"\n📍 Base URL: {BASE_URL}")
    print(f"🔑 API Key: {'✅ Set' if API_KEY else '❌ NOT SET'}")
    print(f"🔒 SSL Verify: {'✅ Enabled' if VERIFY_SSL else '⚠️ Disabled'}")
    
    if not API_KEY:
        print("\n⚠️  ERROR: TCS_GENAI_API_KEY is not set in .env file!")
        return False
    
    # Test models endpoint (if available)
    print(f"\n{'='*60}")
    print("Testing /openai/v1/models endpoint...")
    print("=" * 60)
    
    try:
        with httpx.Client(timeout=30.0, verify=VERIFY_SSL) as client:
            response = client.get(
                f"{BASE_URL}/openai/v1/models",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Available models: {len(data.get('data', []))}")
                for model in data.get("data", [])[:5]:
                    print(f"  - {model.get('id', 'unknown')}")
            else:
                print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")
    
    return True


def test_chat_completion(model: str):
    """Test chat completion with a specific model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print("=" * 60)
    
    endpoint = f"{BASE_URL}/openai/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello' and nothing else."}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        with httpx.Client(timeout=60.0, verify=VERIFY_SSL) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ Response: {content}")
                return True
            else:
                print(f"❌ Error Response: {response.text[:500]}")
                return False
                
    except httpx.TimeoutException:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    """Run all tests."""
    print("\n🔬 Starting TCS GenAI Lab API Diagnostics...\n")
    
    if not test_api_connectivity():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Testing Chat Completions")
    print("=" * 60)
    
    results = {}
    for model in MODELS_TO_TEST:
        results[model] = test_chat_completion(model)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {model}: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! API is working correctly.")
    elif any(results.values()):
        working_models = [m for m, s in results.items() if s]
        print(f"\n⚠️  Some models work. Try using: {working_models[0]}")
        print(f"\nTo change the default model, update TCS_GENAI_MODEL_GPT4O in .env")
    else:
        print("\n❌ All models failed. Possible issues:")
        print("   1. API key may be invalid or expired")
        print("   2. TCS GenAI Lab service may be down")
        print("   3. Network/firewall issues")
        print("   4. Model names may have changed")
        print("\nPlease check with TCS GenAI Lab support.")


if __name__ == "__main__":
    main()
