import os
import pandas as pd
from typing import Any, Dict
from datetime import date, datetime, time
import pytz
from models.options_data import OptionType, OptionsRequest
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.enums import ContractType

OptionsClient = OptionHistoricalDataClient(
    api_key=os.getenv("ALPACA_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY")
)


class OptionsAdapter:
    @staticmethod
    def _build_request_kwargs(req: OptionsRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {"underlying_symbol": req.ticker}

        if req.optionType:
            params["type"] = (
                ContractType.CALL
                if req.optionType == OptionType.CALL
                else ContractType.PUT
            )

        if req.expiry:
            params["expiration_date"] = req.expiry.isoformat()
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

        return params

    @staticmethod
    def _parse_option_to_dict(ticker: str, opt: Any) -> Dict[str, Any] | None:
        symbol = getattr(opt, "symbol", None)
        if not symbol:
            return None

        try:
            root_len = len(ticker)
            if not symbol.startswith(ticker):
                return None

            if len(symbol) < 15 + root_len:
                return None

            suffix = symbol[-15:]
            date_str = suffix[:6]
            type_char = suffix[6]
            strike_str = suffix[7:]

            year = int("20" + date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            exp_date = date(year, month, day)

            opt_type = "CALL" if type_char.upper() == "C" else "PUT"

            strike_price = float(strike_str) / 1000.0

        except (ValueError, IndexError):
            return None

        latest_quote = getattr(opt, "latest_quote", None)
        bid = 0.0
        ask = 0.0
        if latest_quote:
            if isinstance(latest_quote, dict):
                bid = float(latest_quote.get("bid_price", 0) or 0)
                ask = float(latest_quote.get("ask_price", 0) or 0)
            else:
                bid = float(getattr(latest_quote, "bid_price", 0) or 0)
                ask = float(getattr(latest_quote, "ask_price", 0) or 0)

        last_trade = getattr(opt, "latest_trade", None)
        last_price = 0.0
        trade_date = None
        if last_trade:
            if isinstance(last_trade, dict):
                last_price = float(last_trade.get("price", 0) or 0)
                t_date = last_trade.get("timestamp")
                if t_date and hasattr(t_date, "date"):
                    trade_date = t_date.date()
            else:
                last_price = float(getattr(last_trade, "price", 0) or 0)
                t_date = getattr(last_trade, "timestamp", None)
                if t_date and hasattr(t_date, "date"):
                    trade_date = t_date.date()

        mid = (bid + ask) / 2 if (bid and ask) else last_price

        greeks = getattr(opt, "greeks", None)
        implied_vol = None
        delta = None
        gamma = None
        vega = None
        theta = None
        rho = None

        if greeks:
            if isinstance(greeks, dict):
                implied_vol = greeks.get("implied_volatility")
                delta = greeks.get("delta")
                gamma = greeks.get("gamma")
                vega = greeks.get("vega")
                theta = greeks.get("theta")
                rho = greeks.get("rho")
            else:
                implied_vol = getattr(greeks, "implied_volatility", None)
                delta = getattr(greeks, "delta", None)
                gamma = getattr(greeks, "gamma", None)
                vega = getattr(greeks, "vega", None)
                theta = getattr(greeks, "theta", None)
                rho = getattr(greeks, "rho", None)

        return {
            "ticker": ticker,
            "expiry": exp_date,
            "optionType": opt_type,
            "strike": strike_price,
            "lastPrice": last_price,
            "bid": bid,
            "ask": ask,
            "midPrice": mid,
            "volume": 0,
            "openInterest": 0,
            "lastTradeDate": trade_date,
            "impliedVol": float(implied_vol) if implied_vol else None,
            "delta": float(delta) if delta else None,
            "gamma": float(gamma) if gamma else None,
            "vega": float(vega) if vega else None,
            "theta": float(theta) if theta else None,
            "rho": float(rho) if rho else None,
        }

    def fetch_option_chain(self, req: OptionsRequest) -> pd.DataFrame:
        params = self._build_request_kwargs(req)
        response = OptionsClient.get_option_chain(OptionChainRequest(**params))

        if not isinstance(response, dict):
            return pd.DataFrame()

        if req.ticker in response and isinstance(response[req.ticker], (list, dict)):
            raw_chain = response[req.ticker]
            if isinstance(raw_chain, dict):
                raw_chain = raw_chain.values()
        else:
            raw_chain = response.values()

        data_rows = []
        for opt in raw_chain or []:
            parsed = self._parse_option_to_dict(req.ticker, opt)
            if parsed:
                data_rows.append(parsed)
            if len(data_rows) >= req.limit:
                break

        df = pd.DataFrame(data_rows)

        if not df.empty:
            # Timezone handling for accurate Time to Expiry (TTE)
            # US Market Close is generally 16:00 ET.
            # We assume options expire at 16:00 ET on the expiry date.
            et_tz = pytz.timezone("US/Eastern")
            utc_tz = pytz.utc

            # Current time in UTC
            now_utc = datetime.now(utc_tz)

            def calculate_tte(expiry_date):
                # Construct expiry datetime: Expiry Date @ 16:00:00 ET
                # Create naive datetime at 16:00
                expiry_dt_naive = datetime.combine(expiry_date, time(16, 0, 0))
                # Localize to Eastern Time
                expiry_dt_et = et_tz.localize(expiry_dt_naive)
                # Convert to UTC for comparison
                expiry_dt_utc = expiry_dt_et.astimezone(utc_tz)

                # Difference in seconds
                diff_seconds = (expiry_dt_utc - now_utc).total_seconds()

                # Convert to years (365 days * 24 hours * 3600 seconds)
                # Using 365.0 days per year convention
                years = diff_seconds / (365.0 * 24.0 * 3600.0)

                return max(years, 1e-5)  # Ensure strictly positive, minimal TTE

            df["timeToExpiry"] = df["expiry"].apply(calculate_tte)

            # Sort by expiry (ascending) and then by strike (ascending)
            df.sort_values(
                by=["expiry", "strike"], ascending=[True, True], inplace=True
            )
            df.reset_index(drop=True, inplace=True)

        return df
