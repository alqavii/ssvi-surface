import os
import pandas as pd
import numpy as np
from typing import Any, Dict
from datetime import datetime, time, date
import pytz
from models.options_data import OptionType, OptionsRequest
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.enums import ContractType
import yfinance as yf

from adapters.ticker_adapter import TickerAdapter
from engines.zero_rates import ZeroRatesEngine
from utils.tte import tte

OptionsClient = OptionHistoricalDataClient(
    api_key=os.getenv("ALPACA_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY")
)


class OptionsAdapter:
    @staticmethod
    def _build_request_kwargs(
        req: OptionsRequest, spot_price: float, expiry: date = None
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

        # 1. Fast extraction of raw attributes into a DataFrame
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

        # 2. Vectorized Filtering
        root_len = len(ticker)
        mask = (df["symbol"].str.startswith(ticker)) & (
            df["symbol"].str.len() >= 15 + root_len
        )
        df = df[mask].copy()

        if df.empty:
            return pd.DataFrame()

        # 3. Vectorized String Parsing for Expiry, Type, and Strike
        suffix = df["symbol"].str[-15:]
        df["expiry"] = pd.to_datetime(
            "20" + suffix.str[:6], format="%Y%m%d", errors="coerce"
        ).dt.date
        df["optionType"] = suffix.str[6].str.upper().map({"C": "call", "P": "put"})
        df["strike"] = suffix.str[7:].astype(float) / 1000.0

        # 4. Vectorized Attribute Extraction (Quotes, Trades, Greeks)
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

        # Mid Price calculation
        df["midPrice"] = (df["bid"] + df["ask"]) / 2
        df.loc[(df["bid"] == 0) | (df["ask"] == 0), "midPrice"] = df["lastPrice"]

        # Greeks extraction
        greek_fields = {
            "implied_volatility": "impliedVol",
            "delta": "delta",
            "gamma": "gamma",
            "vega": "vega",
            "theta": "theta",
            "rho": "rho",
        }
        for attr, col in greek_fields.items():
            df[col] = df["greeks"].apply(lambda x: get_val(x, attr, None))
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 5. Metadata and Cleanup
        df["ticker"] = ticker
        df["volume"] = 0
        df["openInterest"] = 0

        def extract_date(x):
            ts = get_val(x, "timestamp", None)
            return ts.date() if hasattr(ts, "date") else None

        df["lastTradeDate"] = df["latest_trade"].apply(extract_date)

        return df

    def fetch_option_chain(self, req: OptionsRequest) -> pd.DataFrame:
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
            T = tte([expiry], now_utc=now_utc, market_close_et=time(16, 0, 0))[0]
            rate = ZeroRatesEngine.interpolate_zero_rate(
                pd.DataFrame({"T": [T]}), tte_col="T"
            )[0]
            forward = spot * np.exp((rate - div / 100) * T)

            params = self._build_request_kwargs(req, spot_price=forward, expiry=expiry)
            response = OptionsClient.get_option_chain(OptionChainRequest(**params))

            if not isinstance(response, dict) or not response:
                continue

            # When expiration_date is specified, Alpaca returns a dict of {symbol: contract}
            # instead of {ticker: [contracts]}.
            if req.ticker in response:
                raw_chain = response[req.ticker]
            else:
                # If ticker not in keys, it might be the flat dict of contracts
                raw_chain = list(response.values())

            if isinstance(raw_chain, dict):
                raw_chain = raw_chain.values()

            df_expiry = self._parse_chain_to_df(req.ticker, list(raw_chain or []))
            if not df_expiry.empty:
                df_expiry["timeToExpiry"] = T
                all_dfs.append(df_expiry)

        if not all_dfs:
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)

        if req.limit and not df.empty:
            df = df.head(req.limit)

        if not df.empty:
            # Sort by expiry (ascending) and then by strike (ascending)
            df.sort_values(
                by=["expiry", "strike"], ascending=[True, True], inplace=True
            )
            df.reset_index(drop=True, inplace=True)

        return df
