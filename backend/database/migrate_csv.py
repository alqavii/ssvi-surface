import pandas as pd
from pathlib import Path
from database.config import engine, SessionLocal
from database.models import RiskFreeRate, TreasuryYield, Ticker, SOFRRate

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def migrate_csv_to_database():
    """Migrate all CSV files to database tables"""

    # Create all tables
    from database.models import Base

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Migrate risk_free.csv
        print("Migrating risk_free.csv...")
        risk_free_df = pd.read_csv(DATA_DIR / "risk_free.csv")
        risk_free_df["date"] = pd.to_datetime(risk_free_df["date"]).dt.date

        # Clear existing data
        db.query(RiskFreeRate).delete()

        # Insert new data
        for _, row in risk_free_df.iterrows():
            rate = RiskFreeRate(date=row["date"], value=row["value"])
            db.add(rate)

        print(f"Migrated {len(risk_free_df)} risk-free rate records")

        # Migrate treasury_par_yields.csv
        print("Migrating treasury_par_yields.csv...")
        treasury_df = pd.read_csv(DATA_DIR / "treasury_par_yields.csv")
        treasury_df["date"] = pd.to_datetime(treasury_df["date"]).dt.date

        # Clear existing data
        db.query(TreasuryYield).delete()

        # Insert new data
        for _, row in treasury_df.iterrows():
            yield_data = TreasuryYield(
                date=row["date"],
                maturity_0_5=row.get("0.5"),
                maturity_1_0=row.get("1.0"),
                maturity_2_0=row.get("2.0"),
                maturity_3_0=row.get("3.0"),
                maturity_5_0=row.get("5.0"),
                maturity_7_0=row.get("7.0"),
                maturity_10_0=row.get("10.0"),
            )
            db.add(yield_data)

        print(f"Migrated {len(treasury_df)} treasury yield records")

        # Migrate ticker_universe.csv
        print("Migrating ticker_universe.csv...")
        ticker_df = pd.read_csv(DATA_DIR / "ticker_universe.csv")

        # Clear existing data
        db.query(Ticker).delete()

        # Insert new data
        for _, row in ticker_df.iterrows():
            ticker = Ticker(ticker=row["ticker"], name=row["name"], mcap=row["mcap"])
            db.add(ticker)

        print(f"Migrated {len(ticker_df)} ticker records")

        # Migrate sofr_data.csv
        print("Migrating sofr_data.csv...")
        sofr_df = pd.read_csv(DATA_DIR / "sofr_data.csv")
        sofr_df["date"] = pd.to_datetime(sofr_df["date"]).dt.date

        # Clear existing data
        db.query(SOFRRate).delete()

        # Insert new data
        for _, row in sofr_df.iterrows():
            sofr = SOFRRate(date=row["date"], sofr=row["SOFR"])
            db.add(sofr)

        print(f"Migrated {len(sofr_df)} SOFR records")

        # Commit all changes
        db.commit()
        print("Migration completed successfully!")

    except Exception as e:
        db.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate_csv_to_database()
