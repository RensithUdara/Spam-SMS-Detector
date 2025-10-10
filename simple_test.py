#!/usr/bin/env python3
"""
Simple test to verify the fixed spam detector API
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Spam Detector API")
    print("=" * 40)
    
    # Wait for server to be ready
    print("Waiting for server...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/api/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server not responding")
        return False
    
    # Test cases
    test_cases = [
        {
            "message": "Hello world, how are you?",
            "description": "Normal English message"
        },
        {
            "message": "FREE MONEY! Win $1000 now! Call 1-800-MONEY!",
            "description": "Obvious spam message"
        },
        {
            "message": "අද ඔයාගේ class එකට late වෙලාද?",
            "description": "Sinhala message (the one that caused errors)"
        },
        {
            "message": "",
            "description": "Empty message (should fail gracefully)"
        }
    ]
    
    print(f"\nTesting {len(test_cases)} cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Message: '{test_case['message']}'")
        
        try:
            response = requests.post(
                f"{base_url}/api/predict",
                json={"message": test_case['message']},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'prediction' in data:
                    prediction = data['prediction']
                    confidence = data.get('confidence', 0)
                    print(f"✅ Result: {prediction} (confidence: {confidence:.3f})")
                else:
                    print(f"⚠️ Unexpected response format: {data}")
            elif response.status_code == 400:
                data = response.json()
                print(f"✅ Expected error: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ API error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Request timeout")
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 40)
    print("API test completed!")

if __name__ == "__main__":
    test_api()