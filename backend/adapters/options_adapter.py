import os
import pandas as pd
from typing import List, Any, Dict
from datetime import date
from models.options_data import OptionsModel, OptionType, OptionsRequest
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.enums import ContractType
from utils.time_utils import getTimeToExpiry

OptionsClient = OptionHistoricalDataClient(
    api_key=os.getenv("ALPACA_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY")
)


class OptionsAdapter:
    """Thin wrapper around Alpaca OptionChainRequest -> OptionsModel."""

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
    def _parse_option(ticker: str, opt: Any) -> OptionsModel | None:
        # Try to get details from attributes first, then parse symbol
        symbol = getattr(opt, "symbol", None)
        if not symbol:
            return None

        # Parse symbol if attributes are missing (OptionSnapshot doesn't have contract details)
        # Format: ROOT + YYMMDD + T + SSSSSSSS (OCC format)
        # Example: AAPL251128C00340000
        try:
            # Basic parsing logic assuming standard OCC format
            # Find where the date starts (6 digits)
            # This is a simple parser; might need robustness for different root lengths
            # Assuming root is the ticker provided
            root_len = len(ticker)
            if not symbol.startswith(ticker):
                # Ticker mismatch or non-standard symbol
                return None

            # Extract parts
            # YYMMDD is 6 chars
            # Type is 1 char (C/P)
            # Strike is 8 chars
            # But root length varies.
            # Standard OCC: 6 chars for date, 1 for type, 8 for strike = 15 chars suffix
            if len(symbol) < 15 + root_len:
                # Attempt to parse from end
                # Last 8 is strike
                # Preceding 1 is type
                # Preceding 6 is date
                pass

            suffix = symbol[-15:]
            date_str = suffix[:6]
            type_char = suffix[6]
            strike_str = suffix[7:]

            # Parse Date
            year = int("20" + date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            exp_date = date(year, month, day)

            # Parse Type
            opt_type = OptionType.CALL if type_char.upper() == "C" else OptionType.PUT

            # Parse Strike (divide by 1000)
            strike_price = float(strike_str) / 1000.0

        except (ValueError, IndexError):
            # Fallback if symbol parsing fails
            return None

        # Get market data from snapshot
        latest_quote = getattr(opt, "latest_quote", None)
        bid = 0.0
        ask = 0.0
        if latest_quote:
            # latest_quote might be an object or dict
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
        # Greeks handling (check if dict or object)
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

        return OptionsModel(
            ticker=ticker,
            expiry=exp_date,
            optionType=opt_type,
            strike=strike_price,
            lastPrice=last_price,
            bid=bid,
            ask=ask,
            midPrice=mid,
            volume=0,  # Snapshot doesn't always have volume directly accessible easily
            openInterest=0,  # Snapshot doesn't have OI
            lastTradeDate=trade_date,
            impliedVol=float(implied_vol) if implied_vol else None,
            delta=float(delta) if delta else None,
            gamma=float(gamma) if gamma else None,
            vega=float(vega) if vega else None,
            theta=float(theta) if theta else None,
            rho=float(rho) if rho else None,
        )

    def fetch_option_chain(self, req: OptionsRequest) -> List[OptionsModel]:
        """Fetch contracts from Alpaca and map to OptionsModel."""
        params = self._build_request_kwargs(req)
        response = OptionsClient.get_option_chain(OptionChainRequest(**params))

        # Alpaca returns {symbol: data_dict} directly
        # If specific symbol requested, response might be nested under that symbol,
        # but OptionChainRequest usually returns a flat dict of all contracts.
        # The keys are option symbols like 'AAPL251128P00135000'.
        # Values are the contract details.

        # Check if response is a dict (it should be)
        if not isinstance(response, dict):
            return []

        # If the response is nested under the underlying symbol (unlikely for OptionChainRequest but possible)
        if req.ticker in response and isinstance(response[req.ticker], (list, dict)):
            raw_chain = response[req.ticker]
            # If it's a list, iterate it. If dict, iterate values.
            if isinstance(raw_chain, dict):
                raw_chain = raw_chain.values()
        else:
            # Flat dictionary where keys are option symbols and values are data
            raw_chain = response.values()

        options: List[OptionsModel] = []
        for opt in raw_chain or []:
            parsed = self._parse_option(req.ticker, opt)
            if parsed:
                options.append(parsed)
            if len(options) >= req.limit:
                break

        return getTimeToExpiry(options)

    @staticmethod
    def to_dataframe(options: List[OptionsModel]) -> pd.DataFrame:
        df = pd.DataFrame([opt.model_dump() for opt in options])
        if not df.empty and "optionType" in df.columns:
            df["optionType"] = df["optionType"].apply(
                lambda x: x.value.upper() if hasattr(x, "value") else str(x).upper()
            )
        return df
