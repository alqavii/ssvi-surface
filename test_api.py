#!/usr/bin/env python3
"""
Test script for Petral Trading Dashboard API
Demonstrates how to use the API endpoints and database services
"""

import requests
from datetime import date, timedelta
import sys
from pathlib import Path

# Add backend to path for direct database access
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

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

    # Test rates summary
    print("2. Testing rates summary...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/rates/summary")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(
                f"   Latest SOFR: {data.get('latest_sofr')}% on {data.get('latest_sofr_date')}"
            )
            print(
                f"   Total records: SOFR={data.get('total_sofr_records')}, Treasury={data.get('total_treasury_records')}"
            )
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print()

    # Test risk-free rates
    print("3. Testing risk-free rates...")
    try:
        # Get recent risk-free rates
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        response = requests.get(
            f"{BASE_URL}/api/v1/rates/risk-free",
            params={"start_date": start_date, "end_date": end_date, "limit": 5},
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {len(data)} records")
            if data:
                latest = data[0]
                print(
                    f"   Latest rate: {latest['value'] * 100:.2f}% on {latest['date']}"
                )
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print()

    # Test treasury yields
    print("4. Testing treasury yields...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/rates/treasury", params={"limit": 3}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {len(data)} records")
            if data:
                latest = data[0]
                print(f"   Latest yields on {latest['date']}:")
                print(f"     0.5Y: {latest['maturity_0_5'] * 100:.2f}%")
                print(f"     2.0Y: {latest['maturity_2_0'] * 100:.2f}%")
                print(f"     10Y: {latest['maturity_10_0'] * 100:.2f}%")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print()

    # Test ticker search
    print("5. Testing ticker search...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/tickers", params={"search": "AAPL", "limit": 3}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {len(data)} tickers matching 'AAPL'")
            for ticker in data:
                print(f"   {ticker['ticker']}: {ticker['name']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print()

    # Test risk-free rate calculation
    print("6. Testing risk-free rate calculation...")
    try:
        target_date = date.today() + timedelta(days=90)  # 3 months in the future

        response = requests.get(
            f"{BASE_URL}/api/v1/analytics/risk-free-rate",
            params={"target_date": target_date},
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(
                f"   Risk-free rate for {target_date}: {data['risk_free_rate_percent']:.2f}%"
            )
            print(f"   Calculation method: {data['calculation_method']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n=== API Testing Complete ===")
    return True


def test_database_services():
    """Test database services directly"""
    print("\n=== Testing Database Services Directly ===\n")

    try:
        from database.config import SessionLocal
        from database.services import DatabaseRatesService, DatabaseTickerService

        db = SessionLocal()

        # Test rates service
        print("1. Testing DatabaseRatesService...")
        rates_service = DatabaseRatesService(db)

        # Get latest SOFR rate
        latest_sofr = rates_service.get_sofr_rate(date.today())
        if latest_sofr:
            print(f"   Latest SOFR: {latest_sofr:.2f}%")
        else:
            print("   No SOFR data found")

        # Get latest treasury yields
        latest_yields = rates_service.get_treasury_yields(date.today())
        if latest_yields:
            print(f"   Latest 10Y yield: {latest_yields['10.0'] * 100:.2f}%")
        else:
            print("   No treasury yield data found")

        print()

        # Test ticker service
        print("2. Testing DatabaseTickerService...")
        ticker_service = DatabaseTickerService(db)

        # Search for Apple
        apple_tickers = ticker_service.search_tickers("Apple", limit=3)
        print(f"   Found {len(apple_tickers)} tickers matching 'Apple'")
        for ticker in apple_tickers:
            print(f"   {ticker['ticker']}: {ticker['name']}")

        db.close()
        print("\n=== Database Services Testing Complete ===")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_rates_adapter():
    """Test the database-based rates adapter"""
    print("\n=== Testing Database Rates Adapter ===\n")

    try:
        from adapters.rates_adapter_db import RatesAdapter

        # Test risk-free rate calculation
        print("1. Testing risk-free rate calculation...")
        target_date = date.today()
        rate = RatesAdapter.getRiskFreeRate(target_date)
        if rate:
            print(f"   Risk-free rate for {target_date}: {rate * 100:.2f}%")
        else:
            print(f"   No risk-free rate found for {target_date}")

        # Test future date calculation
        print("2. Testing future date calculation...")
        future_date = date.today() + timedelta(days=180)  # 6 months
        future_rate = RatesAdapter.getRiskFreeRate(future_date)
        if future_rate:
            print(f"   Risk-free rate for {future_date}: {future_rate * 100:.2f}%")
        else:
            print(f"   No risk-free rate found for {future_date}")

        # Test last update dates
        print("3. Testing last update dates...")
        last_update = RatesAdapter.getLastUpdateDate()
        print(f"   Last SOFR update: {last_update.get('sofr')}")
        print(f"   Last treasury update: {last_update.get('treasury')}")

        print("\n=== Rates Adapter Testing Complete ===")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Main test function"""
    print("Petral Trading Dashboard - API and Database Test Suite")
    print("=" * 60)

    # Test API endpoints
    api_success = test_api_endpoints()

    # Test database services
    db_success = test_database_services()

    # Test rates adapter
    adapter_success = test_rates_adapter()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"  API Endpoints: {'✓ PASS' if api_success else '✗ FAIL'}")
    print(f"  Database Services: {'✓ PASS' if db_success else '✗ FAIL'}")
    print(f"  Rates Adapter: {'✓ PASS' if adapter_success else '✗ FAIL'}")

    if api_success and db_success and adapter_success:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure PostgreSQL is running: docker-compose up -d postgres")
        print("2. Make sure the API server is running: python backend/app.py")
        print("3. Make sure data has been migrated: python setup_database.py")


if __name__ == "__main__":
    main()
