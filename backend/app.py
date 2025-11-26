from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
from dateutil.relativedelta import relativedelta

from adapters.options_adapter import OptionsAdapter
from adapters.rates_adapter import RatesAdapter
from adapters.ticker_adapter import TickerAdapter
from engines.IV_smile import IVEngine
from models.options_data import OptionsRequest, OptionType

app = FastAPI(
    title="Petral Trading Dashboard API",
    description="Professional quantitative trading dashboard API with rates, tickers, and analytics",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DashboardRequest(BaseModel):
    ticker: str
    optionType: str = "CALL"
    duration: str = "1y"  # e.g. "1m", "3m", "1y"


def parse_duration(dur: str) -> date:
    today = date.today()
    if dur.endswith("m"):
        months = int(dur[:-1])
        return today + relativedelta(months=months)
    elif dur.endswith("y"):
        years = int(dur[:-1])
        return today + relativedelta(years=years)
    return today + relativedelta(years=1)


@app.post("/api/analytics")
async def get_analytics(req: DashboardRequest):
    try:
        expiry_end = parse_duration(req.duration)

        options_req = OptionsRequest(
            ticker=req.ticker,
            optionType=OptionType(req.optionType.lower()),
            expiryEnd=expiry_end,
            limit=2000,  # Increase limit for better surface resolution
            strikeMin=None,  # Let adapter fetch all relevant strikes
            strikeMax=None,
        )

        adapter = OptionsAdapter()
        df = adapter.fetch_option_chain(options_req)

        if df.empty:
            return {
                "surface": None,
                "smiles": [],
                "candles": [],
                "tickerInfo": {"price": 0, "change": 0, "ivRank": 0},
            }

        # Basic Ticker Info
        base_info = TickerAdapter.fetchBasic(req.ticker)
        rate = RatesAdapter.getTodayRate() or 0.045

        # Calculate IVs
        df = IVEngine.generateIVSmile(df, rate, base_info.dividendYield, base_info.spot)

        # Generate Surface Data
        # Filter out extreme strikes/IVs for cleaner plot
        valid_df = df[
            (df["iv"] > 0.01)
            & (df["iv"] < 5.0)
            & (df["timeToExpiry"] > 0.002)  # Filter very near term noise
        ]

        surface_data = IVEngine.generate_surface_mesh(valid_df, grid_size=30)

        # Generate Smile Curves (Group by Expiry)
        # We take top 5 most liquid expiries or evenly spaced ones
        expiries = sorted(valid_df["expiry"].unique())
        selected_expiries = expiries[:5]  # Take first 5 for now

        smile_data = []
        for exp in selected_expiries:
            slice_df = valid_df[valid_df["expiry"] == exp].sort_values("strike")
            if not slice_df.empty and len(slice_df) > 3:
                tte = slice_df["timeToExpiry"].iloc[0]
                smile_data.append(
                    {
                        "expiry": f"T={tte:.2f}y",
                        "strikes": slice_df["strike"].tolist(),
                        "ivs": slice_df["iv"].tolist(),
                    }
                )

        # Candles (Mock for now, real API usually paid for history)
        # In production, use Alpaca history endpoint here
        candle_data = []

        return {
            "surface": surface_data,
            "smiles": smile_data,
            "candles": candle_data,
            "tickerInfo": {
                "price": base_info.spot,
                "change": 0,  # Calculate from prev close if available
                "ivRank": 50,  # Placeholder, requires 1yr history calculation
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Health"])
async def root():
    return {"message": "Petral Trading Dashboard API", "status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
