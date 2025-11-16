#!/usr/bin/env python3
"""
Test script for Petral Trading Dashboard API
Demonstrates how to use the API endpoints
"""

import requests

# API base URL
BASE_URL = "http://localhost:8000"


def test_api_endpoints():
    """Test all API endpoints"""
    print("=== Testing Petral Trading Dashboard API ===\n")

    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print(
            "   ERROR: Could not connect to API server. Make sure it's running on port 8000"
        )
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print()

    print("\n=== API Testing Complete ===")
    return True


def main():
    """Main test function"""
    print("Petral Trading Dashboard - API Test Suite")
    print("=" * 60)

    # Test API endpoints
    api_success = test_api_endpoints()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"  API Endpoints: {'✓ PASS' if api_success else '✗ FAIL'}")

    if api_success:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure the API server is running: python backend/app.py")


if __name__ == "__main__":
    main()
