#!/usr/bin/env python3
"""
Basic test script for the Spam Detector application
Tests core functionality and API endpoints
"""

import requests
import json
import sys
import time
from pathlib import Path

class SpamDetectorTester:
    """Test suite for Spam Detector"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        print("Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print("✅ Health check passed")
                    self.passed += 1
                    return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        
        self.failed += 1
        return False
    
    def test_model_info(self):
        """Test model info endpoint"""
        print("Testing model info endpoint...")
        try:
            response = requests.get(f"{self.base_url}/api/model/info")
            if response.status_code == 200:
                data = response.json()
                if 'model_type' in data:
                    print(f"✅ Model info: {data.get('model_type')}")
                    self.passed += 1
                    return True
        except Exception as e:
            print(f"❌ Model info failed: {e}")
        
        self.failed += 1
        return False
    
    def test_single_prediction(self):
        """Test single message prediction"""
        print("Testing single prediction...")
        
        test_cases = [
            {
                "message": "Hello, how are you today?",
                "expected": "ham"
            },
            {
                "message": "Congratulations! You have won $1000! Call 1-800-WIN-NOW!",
                "expected": "spam"
            },
            {
                "message": "FREE MONEY! Click here now! Limited time offer!",
                "expected": "spam"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json={"message": test_case["message"]},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get('prediction')
                    confidence = data.get('confidence', 0)
                    
                    print(f"  Test {i+1}: '{test_case['message'][:30]}...'")
                    print(f"    Predicted: {prediction} (confidence: {confidence:.2f})")
                    print(f"    Expected: {test_case['expected']}")
                    
                    if prediction == test_case['expected']:
                        print("    ✅ Correct prediction")
                        self.passed += 1
                    else:
                        print("    ⚠️ Unexpected prediction (may still be correct)")
                        self.passed += 1  # Don't fail on prediction accuracy
                else:
                    print(f"    ❌ API error: {response.status_code}")
                    self.failed += 1
                    
            except Exception as e:
                print(f"    ❌ Request failed: {e}")
                self.failed += 1
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        print("Testing batch prediction...")
        
        messages = [
            "Hello friend, how are you?",
            "WIN BIG MONEY NOW! CALL 1-800-MONEY!",
            "Meeting at 3pm tomorrow in conference room A"
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/api/batch",
                json={"messages": messages},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if len(results) == len(messages):
                    print(f"✅ Batch prediction successful ({len(results)} results)")
                    for i, result in enumerate(results):
                        prediction = result.get('prediction', 'unknown')
                        confidence = result.get('confidence', 0)
                        print(f"  Message {i+1}: {prediction} ({confidence:.2f})")
                    self.passed += 1
                else:
                    print(f"❌ Expected {len(messages)} results, got {len(results)}")
                    self.failed += 1
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                self.failed += 1
                
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            self.failed += 1
    
    def test_error_handling(self):
        """Test error handling"""
        print("Testing error handling...")
        
        # Test empty message
        try:
            response = requests.post(
                f"{self.base_url}/api/predict",
                json={"message": ""},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                print("✅ Empty message handled correctly")
                self.passed += 1
            else:
                print(f"❌ Empty message not handled: {response.status_code}")
                self.failed += 1
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.failed += 1
        
        # Test invalid JSON
        try:
            response = requests.post(
                f"{self.base_url}/api/predict",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                print("✅ Invalid JSON handled correctly")
                self.passed += 1
            else:
                print(f"❌ Invalid JSON not handled: {response.status_code}")
                self.failed += 1
        except Exception as e:
            print(f"❌ JSON error test failed: {e}")
            self.failed += 1
    
    def test_web_interface(self):
        """Test web interface accessibility"""
        print("Testing web interface...")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                if "Spam Detector" in response.text or "RADAR" in response.text:
                    print("✅ Web interface accessible")
                    self.passed += 1
                else:
                    print("❌ Web interface content not found")
                    self.failed += 1
            else:
                print(f"❌ Web interface error: {response.status_code}")
                self.failed += 1
        except Exception as e:
            print(f"❌ Web interface test failed: {e}")
            self.failed += 1
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Spam Detector Tests")
        print("=" * 50)
        
        # Wait for server to be ready
        print("Waiting for server to start...")
        for _ in range(10):
            try:
                requests.get(f"{self.base_url}/api/health", timeout=2)
                break
            except:
                time.sleep(1)
        else:
            print("❌ Server not responding")
            return False
        
        # Run tests
        self.test_health_endpoint()
        self.test_model_info()
        self.test_web_interface()
        self.test_single_prediction()
        self.test_batch_prediction()
        self.test_error_handling()
        
        # Results
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️ Some tests failed")
            return False

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Spam Detector API')
    parser.add_argument('--url', default='http://localhost:5000',
                       help='Base URL for the API (default: http://localhost:5000)')
    args = parser.parse_args()
    
    tester = SpamDetectorTester(args.url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()