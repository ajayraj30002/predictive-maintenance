"""
API Tests for Predictive Maintenance System
Tests all endpoints to ensure they work correctly
"""

import requests
import json
import sys
import time

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

# Sample valid sensor data (based on CMAPSS FD001)
VALID_SAMPLE = {
    "cycle": 150,
    "op_setting_1": -0.0007,
    "op_setting_2": 0.0004,
    "op_setting_3": 100.0,
    "sensor_1": 518.67,
    "sensor_2": 641.45,
    "sensor_3": 1589.70,
    "sensor_4": 1400.18,
    "sensor_5": 14.62,
    "sensor_6": 21.61,
    "sensor_7": 554.36,
    "sensor_8": 2388.06,
    "sensor_9": 9046.19,
    "sensor_10": 1.30,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8138.62,
    "sensor_15": 8.42,
    "sensor_16": 0.03,
    "sensor_17": 392.01,
    "sensor_18": 2388.01,
    "sensor_19": 100.00,
    "sensor_20": 39.06,
    "sensor_21": 23.42
}


def print_test_result(test_name, passed, message=""):
    """Pretty print test results"""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status} - {test_name}")
    if message:
        print(f"     {message}")


def test_health_check():
    """Test root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "status" in data:
                print_test_result("Health Check", True, f"Status: {data['status']}")
                return True
        print_test_result("Health Check", False, f"Status code: {response.status_code}")
        return False
    except Exception as e:
        print_test_result("Health Check", False, str(e))
        return False


def test_features_endpoint():
    """Test /features endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/features", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "feature_names" in data and "num_features" in data:
                print_test_result("Features Endpoint", True, f"Expects {data['num_features']} features")
                return True
        print_test_result("Features Endpoint", False)
        return False
    except Exception as e:
        print_test_result("Features Endpoint", False, str(e))
        return False


def test_valid_prediction():
    """Test prediction with valid data"""
    try:
        response = requests.post(f"{BASE_URL}/predict", json=VALID_SAMPLE, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "rul" in data and "status" in data and "confidence" in data:
                rul = data['rul']
                status = data['status']
                confidence = data['confidence']
                print_test_result("Valid Prediction", True, f"RUL: {rul} cycles, Status: {status}, Confidence: {confidence}")
                return True
        print_test_result("Valid Prediction", False, f"Status code: {response.status_code}")
        return False
    except Exception as e:
        print_test_result("Valid Prediction", False, str(e))
        return False


def test_invalid_data():
    """Test prediction with invalid/missing data (should return 422)"""
    invalid_sample = {"cycle": 150}  # Missing required fields
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_sample, timeout=TIMEOUT)
        if response.status_code == 422:  # Validation error expected
            print_test_result("Invalid Data Handling", True, "Correctly returned 422")
            return True
        print_test_result("Invalid Data Handling", False, f"Expected 422, got {response.status_code}")
        return False
    except Exception as e:
        print_test_result("Invalid Data Handling", False, str(e))
        return False


def test_batch_prediction():
    """Test batch prediction endpoint (if implemented)"""
    # Create a simple CSV in memory
    import io
    import pandas as pd
    
    df = pd.DataFrame([VALID_SAMPLE])
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_batch", 
            files={"file": ("test.csv", csv_buffer, "text/csv")},
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            print_test_result("Batch Prediction", True, f"Processed {data.get('count', 0)} samples")
            return True
        else:
            print_test_result("Batch Prediction", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test_result("Batch Prediction", False, str(e))
        return False


def test_response_time():
    """Test that prediction response is fast (< 200ms)"""
    import time
    
    try:
        start = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=VALID_SAMPLE, timeout=TIMEOUT)
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        
        if response.status_code == 200 and elapsed < 200:
            print_test_result("Response Time", True, f"{elapsed:.1f}ms (<200ms)")
            return True
        elif response.status_code == 200:
            print_test_result("Response Time", False, f"{elapsed:.1f}ms (>200ms)")
            return False
        else:
            print_test_result("Response Time", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_test_result("Response Time", False, str(e))
        return False


def test_cors_headers():
    """Test that CORS headers are present"""
    try:
        response = requests.options(f"{BASE_URL}/predict", timeout=TIMEOUT)
        headers = response.headers
        
        if "access-control-allow-origin" in headers:
            print_test_result("CORS Headers", True, "CORS properly configured")
            return True
        else:
            print_test_result("CORS Headers", False, "No CORS headers found")
            return False
    except Exception as e:
        print_test_result("CORS Headers", False, str(e))
        return False


def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "="*70)
    print("🧪 PREDICTIVE MAINTENANCE API TESTS")
    print("="*70)
    print(f"API URL: {BASE_URL}")
    print("="*70 + "\n")
    
    # Check if API is reachable first
    try:
        requests.get(f"{BASE_URL}/", timeout=5)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API!")
        print(f"   Make sure Docker is running: docker run -p 8000:8000 ajsyraj3002/predictive-maintenance:latest")
        print("   Or run: docker-compose up")
        return False
    
    # Run all tests
    tests = [
        ("Health Check", test_health_check),
        ("Features Endpoint", test_features_endpoint),
        ("Valid Prediction", test_valid_prediction),
        ("Invalid Data Handling", test_invalid_data),
        ("Response Time", test_response_time),
        ("CORS Headers", test_cors_headers),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n▶ Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print("-"*70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! API is ready for deployment.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the logs.")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)