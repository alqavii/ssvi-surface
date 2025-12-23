import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, time as time_cls
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize, brentq
from scipy.interpolate import griddata
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
import os
import pandas as pd
import pytz
from enum import Enum
from typing import Any, Dict, Optional, List, Union, Iterable
from zoneinfo import ZoneInfo
import yfinance as yf
from fredapi import Fred
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.enums import ContractType
import math

# ============================================================================
# INLINED MODELS & HELPER CLASSES
# ============================================================================


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionsRequest:
    def __init__(
        self,
        ticker: str,
        optionType: Optional[OptionType] = None,
        expiry: Optional[date] = None,
        expiryStart: Optional[date] = None,
        expiryEnd: Optional[date] = None,
        strike: Optional[float] = None,
        strikeMin: Optional[float] = None,
        strikeMax: Optional[float] = None,
        moneynessMin: Optional[float] = None,
        moneynessMax: Optional[float] = None,
        limit: Optional[int] = None,
    ):
        self.ticker = ticker
        self.optionType = optionType
        self.expiry = expiry
        self.expiryStart = expiryStart
        self.expiryEnd = expiryEnd
        self.strike = strike
        self.strikeMin = strikeMin
        self.strikeMax = strikeMax
        self.moneynessMin = moneynessMin
        self.moneynessMax = moneynessMax
        self.limit = limit


class TickerModel:
    def __init__(
        self, spot: float, dividendYield: float, exchange: str, timezone: ZoneInfo
    ):
        self.spot = spot
        self.dividendYield = dividendYield
        self.exchange = exchange
        self.timezone = timezone
        self.companyName = None
        self.sector = None
        self.industry = None
        self.marketCap = None
        self.yearHigh = None
        self.yearLow = None


EXCHANGE_TIMEZONES = {
    "NMS": ZoneInfo("America/New_York"),
    "NYQ": ZoneInfo("America/New_York"),
    "PCX": ZoneInfo("America/New_York"),
    "CBOE": ZoneInfo("America/Chicago"),
    "LSE": ZoneInfo("Europe/London"),
    "FRA": ZoneInfo("Europe/Berlin"),
}

TENOR_TO_ID = {
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
}

TBILLS = {1 / 12: "DGS1MO", 0.25: "DGS3MO"}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

ET_TZ = pytz.timezone("US/Eastern")
UTC_TZ = pytz.utc


def tte(
    expiry: List[datetime],
    now_utc: Optional[datetime] = None,
    market_close_et: time_cls = time_cls(16, 0, 0),
    min_tte: float = 1e-5,
):
    """Compute Time-To-Expiry (TTE) in years."""
    if now_utc is None:
        now_utc = datetime.now(UTC_TZ)

    def _one(e: Union[date, datetime]) -> float:
        exp_date = e.date() if isinstance(e, datetime) else e
        expiry_dt_naive = datetime.combine(exp_date, market_close_et)
        expiry_dt_et = ET_TZ.localize(expiry_dt_naive)
        expiry_dt_utc = expiry_dt_et.astimezone(UTC_TZ)
        diff_seconds = (expiry_dt_utc - now_utc).total_seconds()
        years = diff_seconds / (365.0 * 24.0 * 3600.0)
        return max(years, min_tte)

    try:
        if isinstance(expiry, pd.Series):
            return expiry.apply(lambda e: _one(e))
    except Exception:
        pass

    if isinstance(expiry, Iterable) and not isinstance(expiry, (str, bytes)):
        return [_one(e) for e in expiry]

    return _one(expiry)  # type: ignore


def norm_pdf(x):
    """Standard normal probability density function."""
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)


# ============================================================================
# ZERO RATES ENGINE (STATELESS)
# ============================================================================


def calc_zero_rates(yields: pd.Series) -> dict:
    """Calculate discount factors from par yields. Returns dict mapping tenor to discount factor."""
    ys = pd.Series({float(k): float(v) for k, v in yields.items()}).sort_index()  # type: ignore
    discountRates = {}

    if 0.5 in ys.index and not math.isnan(ys.loc[0.5]):
        discountRates[0.5] = 100 / (100 + 100 * ys.loc[0.5] / 2)
    else:
        r = ys.iloc[0]
        discountRates[0.5] = math.exp(-r * 0.5)

    def price_error(x, prev_t, tenor_t, cpn, df_prev):
        known_sum = cpn * sum(
            discountRates[t] for t in discountRates.keys() if t <= prev_t
        )
        steps = int((tenor_t - prev_t) * 2)
        future_sum = 0.0
        for a in range(1, steps):
            future_sum += cpn * df_prev * math.exp(x * -(a * 0.5))
        final_leg = (100 + cpn) * df_prev * math.exp(x * -(tenor_t - prev_t))
        return known_sum + future_sum + final_leg - 100.0

    for tenor, rate in ys.items():
        if tenor in discountRates:
            continue
        prev = max(discountRates.keys())
        c = 100.0 * rate / 2.0

        a, b = -0.5, 5.0
        fa = price_error(a, prev, tenor, c, discountRates[prev])
        fb = price_error(b, prev, tenor, c, discountRates[prev])
        expand = 0
        while fa * fb > 0 and expand < 10:
            a -= 0.5
            b += 0.5
            fa = price_error(a, prev, tenor, c, discountRates[prev])
            fb = price_error(b, prev, tenor, c, discountRates[prev])
            expand += 1

        if fa * fb > 0:
            forwardRate = max(rate, 0.0)
        else:
            forwardRate = brentq(
                lambda x: price_error(x, prev, tenor, c, discountRates[prev]), a, b
            )

        steps = int((tenor - prev) * 2)
        for k in range(1, steps + 1):
            t = prev + k * 0.5
            discountRates[t] = discountRates[prev] * math.exp(-forwardRate * (k * 0.5))  # type: ignore

    desired_bond_tenors = sorted(TENOR_TO_ID.keys())
    filtered = {t: discountRates[t] for t in desired_bond_tenors if t in discountRates}
    return filtered


def fetch_zero_rates_stateless() -> dict:
    """Fetch latest treasury bond and T-bill yields from FRED, calculate zero curve in-memory."""
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error("FRED_API_KEY environment variable not set")
        return {}

    fred = Fred(api_key=fred_key)

    try:
        # Fetch par yields from FRED
        par_yields = {}
        for tenor, series_id in TENOR_TO_ID.items():
            try:
                data = fred.get_series(series_id)
                if not data.empty:
                    latest = float(data.iloc[-1]) / 100.0  # Convert to decimal
                    par_yields[tenor] = latest
            except Exception as e:
                st.warning(f"Could not fetch {series_id}: {e}")

        if not par_yields:
            st.error("Could not fetch any treasury yields")
            return {}

        # Calculate discount factors from par yields
        zero_curve = calc_zero_rates(pd.Series(par_yields))

        # Fetch T-bill rates and add to zero curve
        for tenor, series_id in TBILLS.items():
            try:
                data = fred.get_series(series_id)
                if not data.empty:
                    tbill_rate = float(data.iloc[-1]) / 100.0
                    zero_curve[tenor] = 1.0 / (1.0 + tbill_rate * tenor)
            except Exception as e:
                st.warning(f"Could not fetch T-bill {series_id}: {e}")

        return zero_curve

    except Exception as e:
        st.error(f"Error fetching zero rates: {e}")
        return {}


def interpolate_zero_rate_stateless(
    df,
    tte_col="T",
    zero_curve: dict = None,  # type: ignore
) -> np.ndarray:
    """Interpolate zero rates from in-memory zero curve."""
    if zero_curve is None:
        zero_curve = fetch_zero_rates_stateless()

    if not zero_curve:
        st.error("Could not fetch zero rates")
        return np.full(len(df), np.nan)

    tenors = sorted(zero_curve.keys())
    discount_factors = [zero_curve[t] for t in tenors]

    # Convert discount factors to zero rates: df = exp(-r*t) => r = -ln(df)/t
    log_dfs = np.log(discount_factors)
    zero_rates = -log_dfs / np.array(tenors)

    # Interpolate
    risk_free = np.interp(df[tte_col].values, tenors, zero_rates)
    return risk_free


# ============================================================================
# IV ENGINE
# ============================================================================


class IVEngine:
    @staticmethod
    def _black_scholes(sigma, K, T, r, q, S, is_call):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1 = ndtr(is_call * d1)
        nd2 = ndtr(is_call * d2)

        price = is_call * (S * np.exp(-q * T) * nd1 - K * np.exp(-r * T) * nd2)
        return price

    @staticmethod
    def _vega(sigma, K, T, r, q, S):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        return S * np.exp(-q * T) * norm_pdf(d1) * sqrt_T

    @staticmethod
    def _implied_volatility(
        market_price, K, T, r, q, S, is_call, tolerance=1e-5, max_iter=20
    ):
        sigma = np.full_like(market_price, 0.5, dtype=float)

        for _ in range(max_iter):
            price_est = IVEngine._black_scholes(sigma, K, T, r, q, S, is_call)
            vega = IVEngine._vega(sigma, K, T, r, q, S)

            vega = np.where(vega < 1e-8, 1e-8, vega)

            diff = price_est - market_price

            if np.max(np.abs(diff)) < tolerance:
                break

            sigma = sigma - (diff / vega)
            sigma = np.maximum(sigma, 1e-6)

        final_price = IVEngine._black_scholes(sigma, K, T, r, q, S, is_call)
        final_diff = final_price - market_price

        mask_bad = np.abs(final_diff) > tolerance * 10
        sigma[mask_bad] = np.nan

        return sigma

    @staticmethod
    def generateIVSmile(
        options_df: pd.DataFrame,
        rate: np.ndarray,
        dividendYield: float,
        spot: float,
        optionType: OptionType,
    ) -> pd.DataFrame:
        df = options_df[options_df["optionType"] == optionType.value].copy()
        df = df[["optionType", "strike", "timeToExpiry", "midPrice", "expiry"]]
        df = df.rename(
            columns={
                "optionType": "type",
                "strike": "K",
                "timeToExpiry": "T",
                "midPrice": "Price",
            }
        )
        is_call = np.where(df["type"] == OptionType.CALL.value, 1, -1).astype(float)

        # Extract the rate values for the filtered indices
        rate_filtered = (
            rate[df.index.values]
            if isinstance(rate, np.ndarray)
            else rate.values[df.index.values]
        )

        df["iv"] = IVEngine._implied_volatility(
            df["Price"].values,
            df["K"].values,
            df["T"].values,
            rate_filtered,
            dividendYield,
            spot,
            is_call,
        )
        df["S"] = spot
        df["rate"] = rate_filtered

        return df


# ============================================================================
# TICKER ADAPTER
# ============================================================================


class TickerAdapter:
    @staticmethod
    def fetchBasic(ticker: str) -> TickerModel:
        try:
            # Try to get spot price from Alpaca first
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestQuoteRequest

                client = StockHistoricalDataClient(
                    api_key=os.getenv("ALPACA_KEY"),
                    secret_key=os.getenv("ALPACA_SECRET_KEY"),
                )
                symbol = StockLatestQuoteRequest(symbol_or_symbols=ticker)
                latest_quote = client.get_stock_latest_quote(symbol)
                spot = (
                    latest_quote[ticker].ask_price + latest_quote[ticker].bid_price
                ) / 2
            except Exception:
                spot = None

            # Get from Yahoo Finance
            stock = yf.Ticker(ticker)
            yahoo_spot = stock.history(period="1d")["Close"].iloc[-1]

            # Use Alpaca if available and fresh, otherwise Yahoo
            if spot is None or abs(spot - yahoo_spot) / yahoo_spot > 0.01:
                spot = yahoo_spot

            dividendYield = stock.info.get("dividendYield", 0) or 0
            exchange = stock.info.get("exchange", "N/A")
            timezone = EXCHANGE_TIMEZONES.get(exchange, ZoneInfo("UTC"))

            return TickerModel(
                spot=spot,
                dividendYield=dividendYield,
                exchange=exchange,
                timezone=timezone,
            )
        except Exception as e:
            st.error(f"Error fetching ticker info for {ticker}: {e}")
            # Return default with spot from Yahoo
            stock = yf.Ticker(ticker)
            spot = stock.history(period="1d")["Close"].iloc[-1]
            return TickerModel(
                spot=spot,
                dividendYield=0,
                exchange="N/A",
                timezone=ZoneInfo("UTC"),
            )


# ============================================================================
# OPTIONS ADAPTER
# ============================================================================


class OptionsAdapter:
    @staticmethod
    def _build_request_kwargs(
        req: OptionsRequest,
        spot_price: float,
        expiry: date = None,  # type: ignore
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"underlying_symbol": req.ticker}

        if req.optionType:
            params["type"] = (
                ContractType.CALL
                if req.optionType == OptionType.CALL
                else ContractType.PUT
            )

        target_expiry = expiry or req.expiry
        if target_expiry:
            params["expiration_date"] = target_expiry.isoformat()
        else:
            if req.expiryStart:
                params["expiration_date_gte"] = req.expiryStart.isoformat()
            if req.expiryEnd:
                params["expiration_date_lte"] = req.expiryEnd.isoformat()

        if req.strike:
            params["strike_price_gte"] = req.strike
            params["strike_price_lte"] = req.strike
        else:
            if req.strikeMin is not None:
                params["strike_price_gte"] = req.strikeMin
            if req.strikeMax is not None:
                params["strike_price_lte"] = req.strikeMax
            if req.moneynessMin is not None:
                params["strike_price_gte"] = req.moneynessMin * spot_price
            if req.moneynessMax is not None:
                params["strike_price_lte"] = req.moneynessMax * spot_price

        return params

    @staticmethod
    def _parse_chain_to_df(ticker: str, raw_chain: list) -> pd.DataFrame:
        if not raw_chain:
            return pd.DataFrame()

        data = []
        for opt in raw_chain:
            if isinstance(opt, dict):
                data.append(opt)
            else:
                data.append(
                    {
                        "symbol": getattr(opt, "symbol", None),
                        "latest_quote": getattr(opt, "latest_quote", None),
                        "latest_trade": getattr(opt, "latest_trade", None),
                        "greeks": getattr(opt, "greeks", None),
                    }
                )

        df = pd.DataFrame(data)

        root_len = len(ticker)
        mask = (df["symbol"].str.startswith(ticker)) & (
            df["symbol"].str.len() >= 15 + root_len
        )
        df = df[mask].copy()

        if df.empty:
            return pd.DataFrame()

        suffix = df["symbol"].str[-15:]
        df["expiry"] = pd.to_datetime(
            "20" + suffix.str[:6], format="%Y%m%d", errors="coerce"
        ).dt.date
        df["optionType"] = suffix.str[6].str.upper().map({"C": "call", "P": "put"})
        df["strike"] = suffix.str[7:].astype(float) / 1000.0

        def get_val(obj, field, default=0.0):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(field, default)
            return getattr(obj, field, default)

        df["bid"] = df["latest_quote"].apply(
            lambda x: float(get_val(x, "bid_price") or 0)
        )
        df["ask"] = df["latest_quote"].apply(
            lambda x: float(get_val(x, "ask_price") or 0)
        )
        df["lastPrice"] = df["latest_trade"].apply(
            lambda x: float(get_val(x, "price") or 0)
        )

        df["midPrice"] = (df["bid"] + df["ask"]) / 2
        df.loc[(df["bid"] == 0) | (df["ask"] == 0), "midPrice"] = df["lastPrice"]

        greek_fields = {
            "implied_volatility": "impliedVol",
            "delta": "delta",
            "gamma": "gamma",
            "vega": "vega",
            "theta": "theta",
            "rho": "rho",
        }
        for attr, col in greek_fields.items():
            df[col] = df["greeks"].apply(lambda x: get_val(x, attr, None))  # type: ignore
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["ticker"] = ticker
        df["volume"] = 0
        df["openInterest"] = 0

        def extract_date(x):
            ts = get_val(x, "timestamp", None)  # type: ignore
            return ts.date() if hasattr(ts, "date") else None  # type: ignore

        df["lastTradeDate"] = df["latest_trade"].apply(extract_date)

        return df

    @staticmethod
    def fetch_option_chain(
        req: OptionsRequest,
        zero_curve: dict = None,  # type: ignore
    ) -> pd.DataFrame:
        client = OptionHistoricalDataClient(
            api_key=os.getenv("ALPACA_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
        )

        ticker_info = TickerAdapter.fetchBasic(req.ticker)
        spot = ticker_info.spot
        div = ticker_info.dividendYield

        stock = yf.Ticker(req.ticker)
        if not stock.options:
            return pd.DataFrame()

        all_expiries = [datetime.strptime(e, "%Y-%m-%d").date() for e in stock.options]

        target_expiries = []
        if req.expiry:
            if req.expiry in all_expiries:
                target_expiries = [req.expiry]
        else:
            start = req.expiryStart or date.min
            end = req.expiryEnd or date.max
            target_expiries = [e for e in all_expiries if start <= e <= end]

        if not target_expiries:
            return pd.DataFrame()

        now_utc = datetime.now(pytz.utc)
        all_dfs = []

        for expiry in target_expiries:
            T = tte([expiry], now_utc=now_utc, market_close_et=time_cls(16, 0, 0))[0]  # type: ignore

            # Get zero rate using in-memory zero curve
            if zero_curve is None:
                zero_curve = fetch_zero_rates_stateless()

            tenors = sorted(zero_curve.keys())
            dfs_vals = [zero_curve[t] for t in tenors]
            log_dfs = np.log(dfs_vals)
            zero_rates = -log_dfs / np.array(tenors)
            rate = float(np.interp(T, tenors, zero_rates))

            forward = spot * np.exp((rate - div / 100) * T)

            params = OptionsAdapter._build_request_kwargs(
                req, spot_price=forward, expiry=expiry
            )
            response = client.get_option_chain(OptionChainRequest(**params))

            if not isinstance(response, dict) or not response:
                continue

            if req.ticker in response:
                raw_chain = response[req.ticker]
            else:
                raw_chain = list(response.values())

            if isinstance(raw_chain, dict):
                raw_chain = raw_chain.values()

            df_expiry = OptionsAdapter._parse_chain_to_df(
                req.ticker, list(raw_chain or [])
            )
            if not df_expiry.empty:
                df_expiry["timeToExpiry"] = T
                all_dfs.append(df_expiry)

        if not all_dfs:
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)

        if req.limit and not df.empty:
            df = df.head(req.limit)

        if not df.empty:
            df.sort_values(
                by=["expiry", "strike"], ascending=[True, True], inplace=True
            )
            df.reset_index(drop=True, inplace=True)

        return df


# Set page config
st.set_page_config(page_title="Options IV Analysis", layout="wide")

# Title
st.title("Options Implied Volatility Analysis")
st.markdown("This app analyzes options implied volatility and fits the SSVI model.")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Ticker input
ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA")

# Expiry range selection
expiry_min_weeks = st.sidebar.slider(
    "Min Expiry (weeks)", min_value=0, max_value=52, value=4
)
expiry_max_months = st.sidebar.slider(
    "Max Expiry (months)", min_value=1, max_value=60, value=12
)

# Moneyness range
col1, col2 = st.sidebar.columns(2)
with col1:
    moneyness_min = st.number_input("Moneyness Min", value=0.8, step=0.1)
with col2:
    moneyness_max = st.number_input("Moneyness Max", value=1.2, step=0.1)

# Button to run analysis
run_analysis = st.sidebar.button("Run Analysis", type="primary")

if run_analysis:
    # Add a status container
    status_container = st.container()

    with status_container:
        st.info("Running analysis... This may take a moment.")

        try:
            # Fetch zero rates (stateless, in-memory)
            with st.spinner("Fetching treasury bond yields and building zero curve..."):
                zero_curve = fetch_zero_rates_stateless()

            if not zero_curve:
                st.error("Could not fetch zero rates. Check FRED_API_KEY.")
            else:
                st.success("✓ Fetched zero rates")

                # Setup request
                today = date.today()
                expiry_start = today + relativedelta(weeks=expiry_min_weeks)
                expiry_end = today + relativedelta(months=expiry_max_months)

                req = OptionsRequest(
                    ticker=ticker,
                    optionType=OptionType.CALL,
                    expiryStart=expiry_start,
                    expiryEnd=expiry_end,
                    moneynessMin=moneyness_min,
                    moneynessMax=moneyness_max,
                )

                # Fetch options data
                with st.spinner("Fetching options data..."):
                    df = OptionsAdapter.fetch_option_chain(req, zero_curve=zero_curve)

                st.success(f"Fetched {len(df)} contracts")

                # Calculate IV
                with st.spinner("Calculating implied volatility..."):
                    # Interpolate zero rates for the dataframe
                    rate_array = interpolate_zero_rate_stateless(
                        df, tte_col="timeToExpiry", zero_curve=zero_curve
                    )

                    base_info = TickerAdapter.fetchBasic(req.ticker)
                    div = base_info.dividendYield
                    spot = base_info.spot

                    # Generate IV smile - rate will be included in the returned dataframe
                    surface_data = IVEngine.generateIVSmile(
                        df, rate_array, div / 100, spot, OptionType.CALL
                    )
                    surface_data.dropna(inplace=True)

                st.success(f"✓ Calculated IV for {len(surface_data)} data points")

                # Display basic info
                col1, col2, col3 = st.columns(3)
                col1.metric("Spot Price", f"${spot:.2f}")
                col2.metric("Dividend Yield", f"{div:.2f}%")
                col3.metric("Data Points", len(surface_data))

                # Prepare SSVI data
                surface_data["F"] = spot * np.exp(
                    (surface_data["rate"] - div / 100) * surface_data["T"]
                )
                surface_data["k"] = np.log(surface_data["K"] / surface_data["F"])
                surface_data["w"] = surface_data["iv"] ** 2 * surface_data["T"]

                # Calculate theta

                thetas = surface_data.loc[
                    surface_data["k"].abs().groupby(surface_data["T"]).idxmin()
                ][["T", "k", "w"]].reset_index(drop=True)

                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                theta_iso = iso.fit_transform(thetas["T"], thetas["w"])  # type: ignore

                theta_spline = PchipInterpolator(
                    thetas["T"], theta_iso, extrapolate=True
                )
                surface_data["theta"] = theta_spline(surface_data["T"])
                # Calculate vega
                vega = IVEngine._vega(
                    surface_data["iv"],
                    surface_data["K"],
                    surface_data["T"],
                    surface_data["rate"],
                    div,
                    surface_data["F"],
                )
                surface_data["vega"] = np.clip(vega, 1e-6, None)

                # SSVI calibration
                with st.spinner("Calibrating SSVI model..."):
                    T_vals = surface_data["T"].values
                    T_unique = np.sort(surface_data["T"].unique())
                    T_to_index = {t: i for i, t in enumerate(T_unique)}
                    idx = np.array([T_to_index[t] for t in T_vals])
                    x0 = np.r_[-0.3, np.full(len(T_unique), 0.5)]

                    def ssvi_w(k, theta, phi, rho):
                        w_ssvi = (
                            1
                            / 2
                            * theta
                            * (
                                1
                                + rho * phi * k
                                + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2)
                            )
                        )
                        return w_ssvi

                    weights = (
                        surface_data.groupby("T")["vega"].transform(
                            lambda x: x / x.sum()
                        )
                        * 100
                    )

                    def objective(x, theta, k, w_mkt, idx):
                        rho_raw = x[0]
                        rho = np.tanh(rho_raw)
                        eta = x[1:]
                        phi = eta[idx] / np.sqrt(np.maximum(theta, 1e-12))
                        w_model = ssvi_w(k, theta, phi, rho)
                        error = w_model - w_mkt
                        loss = np.dot(weights * error, error)
                        return loss

                    def make_constraints(n_expiries):
                        cons = []
                        for i in range(n_expiries):
                            cons.append(
                                {
                                    "type": "ineq",
                                    "fun": lambda x, i=i: 2
                                    - x[1 + i] * (1 + abs(x[0])),
                                }
                            )
                        return cons

                    bounds = [(-5.0, 5.0)] + [(1e-5, 2)] * len(T_unique)
                    w_mkt = surface_data["w"].values
                    k = surface_data["k"].values
                    theta = surface_data["theta"].values

                    res = minimize(
                        objective,
                        x0=x0,
                        args=(theta, k, w_mkt, idx),
                        method="SLSQP",
                        bounds=bounds,
                        constraints=make_constraints(len(T_unique)),
                    )

                    rho_opt = np.tanh(res.x[0])
                    eta_opt = res.x[1:]
                    ssvi_loss = objective(res.x, theta, k, w_mkt, idx)

                    surface_data["w_ssvi"] = ssvi_w(
                        k,
                        theta,
                        phi=eta_opt[idx] / np.sqrt(np.maximum(theta, 1e-12)),  # type: ignore
                        rho=rho_opt,
                    )

                st.success("✓ SSVI calibration complete")

                # Display SSVI parameters
                col1, col2 = st.columns(2)
                col1.metric("Optimized Rho", f"{rho_opt:.6f}")
                col2.metric("SSVI Loss", f"{ssvi_loss:.6f}")

                # Calculate errors
                surface_data["residuals"] = surface_data["w_ssvi"] - surface_data["w"]
                surface_data["relative_residuals"] = (
                    surface_data["residuals"] / surface_data["w"]
                ) * 100

                surface_data["iv_ssvi"] = np.sqrt(
                    surface_data["w_ssvi"] / surface_data["T"]
                )
                surface_data["iv_error"] = (
                    surface_data["iv_ssvi"] / surface_data["iv"] - 1
                ) * 100

                # Display data tables
                st.header("Data Summary")

                tab1, tab2, tab3 = st.tabs(
                    ["Options Data", "Surface Data", "SSVI Results"]
                )

                with tab1:
                    st.dataframe(
                        df[
                            [
                                "optionType",
                                "strike",
                                "timeToExpiry",
                                "midPrice",
                                "expiry",
                            ]
                        ],
                        use_container_width=True,
                        height=400,
                    )

                with tab2:
                    st.dataframe(
                        surface_data[["expiry", "K", "k", "w", "theta"]],
                        use_container_width=True,
                        height=400,
                    )

                with tab3:
                    st.dataframe(
                        surface_data[["expiry", "K", "iv", "iv_ssvi", "iv_error"]],
                        use_container_width=True,
                        height=400,
                    )

                # Visualizations
                st.header("Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("SSVI Residuals")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(
                        surface_data["k"],
                        surface_data["residuals"],
                        c=surface_data["T"],
                        cmap="viridis",
                        alpha=0.6,
                        s=50,
                    )
                    plt.colorbar(scatter, ax=ax, label="Time to Expiry (T)")
                    ax.set_xlabel("Log-Moneyness (k)")
                    ax.set_ylabel("Residuals")
                    ax.set_title("SSVI Model Residuals vs Log-Moneyness")
                    ax.axhline(0, color="red", linestyle="--", linewidth=2)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.subheader("IV Error (%)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(
                        surface_data["k"],
                        surface_data["iv_error"],
                        c=surface_data["T"],
                        cmap="viridis",
                        alpha=0.6,
                        s=50,
                    )
                    plt.colorbar(scatter, ax=ax, label="Time to Expiry (T)")
                    ax.set_xlabel("Log-Moneyness (k)")
                    ax.set_ylabel("Implied Volatility Error (%)")
                    ax.set_title("IV Error vs Log-Moneyness")
                    ax.axhline(0, color="red", linestyle="--", linewidth=2)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()

                # Additional metrics
                st.header("Performance Metrics")

                mean_iv_error = np.mean(np.abs(surface_data["iv_error"]))
                max_iv_error = np.max(np.abs(surface_data["iv_error"]))
                rmse = np.sqrt(
                    np.mean((surface_data["iv_ssvi"] - surface_data["iv"]) ** 2)
                )
                vega_weighted_rmse = np.sqrt(
                    np.sum(
                        ((surface_data["iv_ssvi"] - surface_data["iv"]) ** 2)
                        * (surface_data["vega"] / np.sum(surface_data["vega"]))
                    )
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("IV RMSE", f"{rmse * 100:.4f}%")
                col2.metric("Vega Weighted RMSE", f"{vega_weighted_rmse * 100:.4f}%")
                col3.metric("Mean IV Error", f"{mean_iv_error:.4f}%")
                col4.metric("Max IV Error", f"{max_iv_error:.4f}%")

                st.success("✓ Analysis complete!")

                # 3D Surface Plots - Market vs Model
                st.header("IV Surface Comparison")

                # Prepare data for 3D surface plots
                # Create a regular grid for interpolation
                k_min, k_max = surface_data["k"].min(), surface_data["k"].max()
                T_min, T_max = surface_data["T"].min(), surface_data["T"].max()

                # Create a grid with 50x50 points
                grid_k, grid_T = np.mgrid[k_min:k_max:50j, T_min:T_max:50j]

                # Convert log-moneyness to moneyness (K/F) for plotting
                grid_M = np.exp(grid_k)

                # Interpolate data onto the grid
                points = surface_data[["k", "T"]].values

                # Market IV interpolation
                grid_iv_market = griddata(
                    points,
                    surface_data["iv"].values,
                    (grid_k, grid_T),
                    method="linear",
                )

                # Model IV interpolation
                grid_iv_model = griddata(
                    points,
                    surface_data["iv_ssvi"].values,
                    (grid_k, grid_T),
                    method="linear",
                )

                # Create side-by-side 3D surface plots
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "surface"}, {"type": "surface"}]],
                    subplot_titles=("Market IV Surface", "Model IV Surface"),
                )

                # Market IV surface
                fig.add_trace(
                    go.Surface(
                        x=grid_M,
                        y=grid_T,
                        z=grid_iv_market,
                        colorscale="Viridis",
                        name="Market IV",
                        showscale=False,
                    ),
                    row=1,
                    col=1,
                )

                # Model IV surface
                fig.add_trace(
                    go.Surface(
                        x=grid_M,
                        y=grid_T,
                        z=grid_iv_model,
                        colorscale="Viridis",
                        name="Model IV",
                        showscale=True,
                        colorbar=dict(x=1.02, len=0.8),
                    ),
                    row=1,
                    col=2,
                )

                # Update layout
                fig.update_layout(
                    title_text="Implied Volatility Surface: Market vs SSVI Model",
                    height=700,
                    width=1400,
                    showlegend=False,
                    scene=dict(
                        xaxis_title="Moneyness (K/F)",
                        yaxis_title="Time to Expiry (Years)",
                        zaxis_title="Implied Volatility",
                    ),
                    scene2=dict(
                        xaxis_title="Moneyness (K/F)",
                        yaxis_title="Time to Expiry (Years)",
                        zaxis_title="Implied Volatility",
                    ),
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback

            st.error(traceback.format_exc())
else:
    st.info(
        "<-- Configure the parameters in the sidebar and click 'Run Analysis' to begin."
    )
