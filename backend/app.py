from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel
from database.config import get_db
from database.services import DatabaseRatesService, DatabaseTickerService
from database.models import RiskFreeRate, TreasuryYield, Ticker, SOFRRate

# Create FastAPI app
app = FastAPI(
    title="Petral Trading Dashboard API",
    description="Professional quantitative trading dashboard API with rates, tickers, and analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API responses
class RiskFreeRateResponse(BaseModel):
    id: int
    date: date
    value: float
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TreasuryYieldResponse(BaseModel):
    id: int
    date: date
    maturity_0_5: Optional[float] = None
    maturity_1_0: Optional[float] = None
    maturity_2_0: Optional[float] = None
    maturity_3_0: Optional[float] = None
    maturity_5_0: Optional[float] = None
    maturity_7_0: Optional[float] = None
    maturity_10_0: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TickerResponse(BaseModel):
    id: int
    ticker: str
    name: Optional[str] = None
    mcap: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SOFRRateResponse(BaseModel):
    id: int
    date: date
    sofr: float
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RatesSummaryResponse(BaseModel):
    latest_sofr: Optional[float] = None
    latest_sofr_date: Optional[date] = None
    latest_treasury_date: Optional[date] = None
    total_risk_free_records: int
    total_treasury_records: int
    total_ticker_records: int
    total_sofr_records: int


# API Routes


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"message": "Petral Trading Dashboard API", "status": "healthy"}


@app.get("/api/v1/rates/summary", response_model=RatesSummaryResponse, tags=["Rates"])
async def get_rates_summary(db: Session = Depends(get_db)):
    """
    Get a summary of all rates data in the database.

    Returns:
    - Latest SOFR rate and date
    - Latest treasury yield date
    - Total record counts for each table
    """
    try:
        # Get latest SOFR
        latest_sofr = db.query(SOFRRate).order_by(SOFRRate.date.desc()).first()

        # Get latest treasury yield date
        latest_treasury = (
            db.query(TreasuryYield).order_by(TreasuryYield.date.desc()).first()
        )

        # Get counts
        risk_free_count = db.query(RiskFreeRate).count()
        treasury_count = db.query(TreasuryYield).count()
        ticker_count = db.query(Ticker).count()
        sofr_count = db.query(SOFRRate).count()

        return RatesSummaryResponse(
            latest_sofr=latest_sofr.sofr if latest_sofr else None,
            latest_sofr_date=latest_sofr.date if latest_sofr else None,
            latest_treasury_date=latest_treasury.date if latest_treasury else None,
            total_risk_free_records=risk_free_count,
            total_treasury_records=treasury_count,
            total_ticker_records=ticker_count,
            total_sofr_records=sofr_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/rates/risk-free", response_model=List[RiskFreeRateResponse], tags=["Rates"]
)
async def get_risk_free_rates(
    start_date: Optional[date] = Query(None, description="Start date for filtering"),
    end_date: Optional[date] = Query(None, description="End date for filtering"),
    limit: int = Query(100, description="Maximum number of records to return"),
    db: Session = Depends(get_db),
):
    """
    Get risk-free rates data.

    Parameters:
    - start_date: Filter records from this date onwards
    - end_date: Filter records up to this date
    - limit: Maximum number of records to return (default: 100)

    Returns:
    List of risk-free rate records
    """
    try:
        query = db.query(RiskFreeRate)

        if start_date:
            query = query.filter(RiskFreeRate.date >= start_date)
        if end_date:
            query = query.filter(RiskFreeRate.date <= end_date)

        rates = query.order_by(RiskFreeRate.date.desc()).limit(limit).all()
        return rates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rates/risk-free/{target_date}", tags=["Rates"])
async def get_risk_free_rate_for_date(target_date: date, db: Session = Depends(get_db)):
    """
    Get the risk-free rate for a specific date.

    Returns the latest available rate on or before the target date.
    """
    try:
        service = DatabaseRatesService(db)
        rate = service.get_risk_free_rate(target_date)

        if rate is None:
            raise HTTPException(
                status_code=404, detail=f"No risk-free rate found for {target_date}"
            )

        return {"date": target_date, "rate": rate, "rate_percent": rate * 100}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/rates/treasury", response_model=List[TreasuryYieldResponse], tags=["Rates"]
)
async def get_treasury_yields(
    start_date: Optional[date] = Query(None, description="Start date for filtering"),
    end_date: Optional[date] = Query(None, description="End date for filtering"),
    limit: int = Query(100, description="Maximum number of records to return"),
    db: Session = Depends(get_db),
):
    """
    Get treasury yield data for different maturities.

    Parameters:
    - start_date: Filter records from this date onwards
    - end_date: Filter records up to this date
    - limit: Maximum number of records to return (default: 100)

    Returns:
    List of treasury yield records with maturities: 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0 years
    """
    try:
        query = db.query(TreasuryYield)

        if start_date:
            query = query.filter(TreasuryYield.date >= start_date)
        if end_date:
            query = query.filter(TreasuryYield.date <= end_date)

        yields = query.order_by(TreasuryYield.date.desc()).limit(limit).all()
        return yields
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rates/treasury/{target_date}", tags=["Rates"])
async def get_treasury_yields_for_date(
    target_date: date, db: Session = Depends(get_db)
):
    """
    Get treasury yields for a specific date.

    Returns yields for all maturities (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0 years)
    for the latest available date on or before the target date.
    """
    try:
        service = DatabaseRatesService(db)
        yields = service.get_treasury_yields(target_date)

        if yields is None:
            raise HTTPException(
                status_code=404, detail=f"No treasury yields found for {target_date}"
            )

        return {"date": target_date, "yields": yields}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rates/sofr", response_model=List[SOFRRateResponse], tags=["Rates"])
async def get_sofr_rates(
    start_date: Optional[date] = Query(None, description="Start date for filtering"),
    end_date: Optional[date] = Query(None, description="End date for filtering"),
    limit: int = Query(100, description="Maximum number of records to return"),
    db: Session = Depends(get_db),
):
    """
    Get SOFR (Secured Overnight Financing Rate) data.

    Parameters:
    - start_date: Filter records from this date onwards
    - end_date: Filter records up to this date
    - limit: Maximum number of records to return (default: 100)

    Returns:
    List of SOFR rate records
    """
    try:
        query = db.query(SOFRRate)

        if start_date:
            query = query.filter(SOFRRate.date >= start_date)
        if end_date:
            query = query.filter(SOFRRate.date <= end_date)

        rates = query.order_by(SOFRRate.date.desc()).limit(limit).all()
        return rates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tickers", response_model=List[TickerResponse], tags=["Tickers"])
async def get_tickers(
    search: Optional[str] = Query(
        None, description="Search by ticker symbol or company name"
    ),
    limit: int = Query(100, description="Maximum number of records to return"),
    db: Session = Depends(get_db),
):
    """
    Get ticker information.

    Parameters:
    - search: Search by ticker symbol or company name (case-insensitive)
    - limit: Maximum number of records to return (default: 100)

    Returns:
    List of ticker records
    """
    try:
        service = DatabaseTickerService(db)

        if search:
            tickers = service.search_tickers(search, limit)
        else:
            tickers = service.get_all_tickers()[:limit]

        return tickers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tickers/{symbol}", tags=["Tickers"])
async def get_ticker_by_symbol(symbol: str, db: Session = Depends(get_db)):
    """
    Get ticker information by symbol.

    Parameters:
    - symbol: Ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
    Ticker information including name and market cap
    """
    try:
        service = DatabaseTickerService(db)
        ticker = service.get_ticker(symbol.upper())

        if ticker is None:
            raise HTTPException(status_code=404, detail=f"Ticker '{symbol}' not found")

        return ticker
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/risk-free-rate", tags=["Analytics"])
async def calculate_risk_free_rate(
    target_date: date,
    request_date: Optional[date] = Query(
        None, description="Request date for calculation (defaults to today)"
    ),
    db: Session = Depends(get_db),
):
    """
    Calculate risk-free rate for a target date using the same logic as RatesAdapter.

    For past dates: Uses SOFR rate
    For future dates: Uses bootstrapped zero rates from treasury yields

    Parameters:
    - target_date: Date for which to calculate the risk-free rate
    - request_date: Reference date for calculation (defaults to today)

    Returns:
    Risk-free rate as decimal and percentage
    """
    try:
        from adapters.rates_adapter_db import RatesAdapter

        rate = RatesAdapter.getRiskFreeRate(target_date, request_date)

        if rate is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not calculate risk-free rate for {target_date}",
            )

        return {
            "target_date": target_date,
            "request_date": request_date or datetime.today().date(),
            "risk_free_rate": rate,
            "risk_free_rate_percent": rate * 100,
            "calculation_method": "SOFR"
            if target_date <= (request_date or datetime.today().date())
            else "Bootstrapped Zero Curve",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
