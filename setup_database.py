#!/usr/bin/env python3
"""
Database setup and migration script for Petral trading dashboard.
This script will:
1. Start PostgreSQL database (if using Docker)
2. Create database tables
3. Migrate CSV data to database
4. Verify the migration
"""

import sys
import subprocess
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))


def check_docker():
    """Check if Docker is running"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def start_postgres():
    """Start PostgreSQL using Docker Compose"""
    if not check_docker():
        print("Docker not found. Please install Docker or set up PostgreSQL manually.")
        return False

    print("Starting PostgreSQL with Docker Compose...")
    try:
        subprocess.run(["docker-compose", "up", "-d", "postgres"], check=True)
        print("PostgreSQL started successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to start PostgreSQL: {e}")
        return False


def wait_for_postgres():
    """Wait for PostgreSQL to be ready"""
    import time

    print("Waiting for PostgreSQL to be ready...")
    time.sleep(10)  # Give PostgreSQL time to start


def run_migration():
    """Run the CSV to database migration"""
    print("Running CSV to database migration...")
    try:
        from database.migrate_csv import migrate_csv_to_database

        migrate_csv_to_database()
        return True
    except Exception as e:
        print(f"Migration failed: {e}")
        return False


def verify_migration():
    """Verify that the migration was successful"""
    print("Verifying migration...")
    try:
        from database.config import SessionLocal
        from database.models import RiskFreeRate, TreasuryYield, Ticker, SOFRRate

        db = SessionLocal()
        try:
            # Check record counts
            risk_free_count = db.query(RiskFreeRate).count()
            treasury_count = db.query(TreasuryYield).count()
            ticker_count = db.query(Ticker).count()
            sofr_count = db.query(SOFRRate).count()

            print("Migration verification:")
            print(f"  Risk-free rates: {risk_free_count} records")
            print(f"  Treasury yields: {treasury_count} records")
            print(f"  Tickers: {ticker_count} records")
            print(f"  SOFR rates: {sofr_count} records")

            # Test a query
            latest_sofr = db.query(SOFRRate).order_by(SOFRRate.date.desc()).first()
            if latest_sofr:
                print(f"  Latest SOFR: {latest_sofr.sofr}% on {latest_sofr.date}")

            return True
        finally:
            db.close()
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("=== Petral Database Setup ===")

    # Check if we should start PostgreSQL
    if len(sys.argv) > 1 and sys.argv[1] == "--start-db":
        if not start_postgres():
            sys.exit(1)
        wait_for_postgres()

    # Run migration
    if not run_migration():
        sys.exit(1)

    # Verify migration
    if not verify_migration():
        sys.exit(1)

    print("\n=== Database Setup Complete ===")
    print("You can now use the database-based adapters!")
    print("\nTo use the new database adapter:")
    print("  from adapters.rates_adapter_db import RatesAdapter")
    print("\nTo start PostgreSQL in the future:")
    print("  docker-compose up -d postgres")


if __name__ == "__main__":
    main()
