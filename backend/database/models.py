from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Index
from sqlalchemy.sql import func
from database.config import Base


class RiskFreeRate(Base):
    """Risk-free rate data (SOFR)"""

    __tablename__ = "risk_free_rates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    value = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (Index("idx_risk_free_date", "date"),)


class TreasuryYield(Base):
    """Treasury par yields for different maturities"""

    __tablename__ = "treasury_yields"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    maturity_0_5 = Column(Float, nullable=True)  # 0.5 year
    maturity_1_0 = Column(Float, nullable=True)  # 1.0 year
    maturity_2_0 = Column(Float, nullable=True)  # 2.0 year
    maturity_3_0 = Column(Float, nullable=True)  # 3.0 year
    maturity_5_0 = Column(Float, nullable=True)  # 5.0 year
    maturity_7_0 = Column(Float, nullable=True)  # 7.0 year
    maturity_10_0 = Column(Float, nullable=True)  # 10.0 year
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (Index("idx_treasury_date", "date"),)


class Ticker(Base):
    """Ticker universe data"""

    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(Text, nullable=True)
    mcap = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (Index("idx_ticker_symbol", "ticker"),)


class SOFRRate(Base):
    """SOFR (Secured Overnight Financing Rate) data"""

    __tablename__ = "sofr_rates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    sofr = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (Index("idx_sofr_date", "date"),)
