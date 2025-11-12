from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from datetime import date
from typing import List, Optional, Dict, Any
from database.models import RiskFreeRate, TreasuryYield, Ticker, SOFRRate


class DatabaseRatesService:
    """Service class for accessing rates data from database"""

    def __init__(self, db: Session):
        self.db = db

    def get_risk_free_rate(self, target_date: date) -> Optional[float]:
        """Get risk-free rate for a specific date"""
        rate = (
            self.db.query(RiskFreeRate)
            .filter(RiskFreeRate.date <= target_date)
            .order_by(desc(RiskFreeRate.date))
            .first()
        )

        return rate.value if rate else None

    def get_risk_free_rates_range(
        self, start_date: date, end_date: date
    ) -> List[Dict[str, Any]]:
        """Get risk-free rates for a date range"""
        rates = (
            self.db.query(RiskFreeRate)
            .filter(
                and_(RiskFreeRate.date >= start_date, RiskFreeRate.date <= end_date)
            )
            .order_by(RiskFreeRate.date)
            .all()
        )

        return [{"date": rate.date, "value": rate.value} for rate in rates]

    def get_treasury_yields(self, target_date: date) -> Optional[Dict[str, float]]:
        """Get treasury yields for a specific date"""
        yields = (
            self.db.query(TreasuryYield)
            .filter(TreasuryYield.date <= target_date)
            .order_by(desc(TreasuryYield.date))
            .first()
        )

        if not yields:
            return None

        return {
            "0.5": yields.maturity_0_5,
            "1.0": yields.maturity_1_0,
            "2.0": yields.maturity_2_0,
            "3.0": yields.maturity_3_0,
            "5.0": yields.maturity_5_0,
            "7.0": yields.maturity_7_0,
            "10.0": yields.maturity_10_0,
        }

    def get_treasury_yields_range(
        self, start_date: date, end_date: date
    ) -> List[Dict[str, Any]]:
        """Get treasury yields for a date range"""
        yields = (
            self.db.query(TreasuryYield)
            .filter(
                and_(TreasuryYield.date >= start_date, TreasuryYield.date <= end_date)
            )
            .order_by(TreasuryYield.date)
            .all()
        )

        return [
            {
                "date": yield_data.date,
                "0.5": yield_data.maturity_0_5,
                "1.0": yield_data.maturity_1_0,
                "2.0": yield_data.maturity_2_0,
                "3.0": yield_data.maturity_3_0,
                "5.0": yield_data.maturity_5_0,
                "7.0": yield_data.maturity_7_0,
                "10.0": yield_data.maturity_10_0,
            }
            for yield_data in yields
        ]

    def get_sofr_rate(self, target_date: date) -> Optional[float]:
        """Get SOFR rate for a specific date"""
        rate = (
            self.db.query(SOFRRate)
            .filter(SOFRRate.date <= target_date)
            .order_by(desc(SOFRRate.date))
            .first()
        )

        return rate.sofr if rate else None

    def get_sofr_rates_range(
        self, start_date: date, end_date: date
    ) -> List[Dict[str, Any]]:
        """Get SOFR rates for a date range"""
        rates = (
            self.db.query(SOFRRate)
            .filter(and_(SOFRRate.date >= start_date, SOFRRate.date <= end_date))
            .order_by(SOFRRate.date)
            .all()
        )

        return [{"date": rate.date, "SOFR": rate.sofr} for rate in rates]


class DatabaseTickerService:
    """Service class for accessing ticker data from database"""

    def __init__(self, db: Session):
        self.db = db

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker information by symbol"""
        ticker = self.db.query(Ticker).filter(Ticker.ticker == symbol).first()

        if not ticker:
            return None

        return {"ticker": ticker.ticker, "name": ticker.name, "mcap": ticker.mcap}

    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """Get all tickers"""
        tickers = self.db.query(Ticker).all()

        return [
            {"ticker": ticker.ticker, "name": ticker.name, "mcap": ticker.mcap}
            for ticker in tickers
        ]

    def search_tickers(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search tickers by symbol or name"""
        tickers = (
            self.db.query(Ticker)
            .filter(Ticker.ticker.ilike(f"%{query}%") | Ticker.name.ilike(f"%{query}%"))
            .limit(limit)
            .all()
        )

        return [
            {"ticker": ticker.ticker, "name": ticker.name, "mcap": ticker.mcap}
            for ticker in tickers
        ]
