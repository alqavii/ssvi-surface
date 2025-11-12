import os
import pandas as pd
from typing import List, Any
from models.config_model import Config
from models.ticker_data import TickerModel
from models.options_data import OptionsModel, OptionType
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

OptionsClient = OptionHistoricalDataClient(
    api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY
)


class OptionsAdapter:
    @staticmethod
    def fetchOptionChain(cfg: Config, ticker: TickerModel) -> List[OptionsModel]:
        request = OptionChainRequest(underlying_symbol=cfg.ticker)  # type: ignore
        chain = OptionsClient.get_option_chain(request)

        options = []
        chain_data: Any = chain.get(cfg.ticker) if isinstance(chain, dict) else chain

        for opt in chain_data or []:
            opt_type = (
                OptionType.CALL
                if getattr(opt, "option_type", "").lower() == "call"
                else OptionType.PUT
            )
            bid = getattr(opt, "bid", None) or 0
            ask = getattr(opt, "ask", None) or 0
            mid = (
                (bid + ask) / 2
                if bid and ask
                else getattr(opt, "last_price", None) or 0
            )
            greeks = getattr(opt, "greeks", None)

            exp_date = getattr(opt, "expiration_date", None)
            if exp_date and hasattr(exp_date, "date"):
                exp_date = exp_date.date()
            if not exp_date:
                continue

            trade_date = getattr(opt, "last_trade_date", None)
            if trade_date and hasattr(trade_date, "date"):
                trade_date = trade_date.date()

            options.append(
                OptionsModel(
                    ticker=cfg.ticker,
                    expiry=exp_date,
                    optionType=opt_type,
                    strike=float(getattr(opt, "strike_price", 0)),
                    lastPrice=float(getattr(opt, "last_price", 0)),
                    bid=float(bid),
                    ask=float(ask),
                    midPrice=mid,
                    volume=getattr(opt, "volume", None),
                    openInterest=getattr(opt, "open_interest", None),
                    lastTradeDate=trade_date,  # type: ignore
                    impliedVol=getattr(opt, "implied_volatility", None),
                    delta=getattr(greeks, "delta", None) if greeks else None,
                    gamma=getattr(greeks, "gamma", None) if greeks else None,
                    vega=getattr(greeks, "vega", None) if greeks else None,
                    theta=getattr(greeks, "theta", None) if greeks else None,
                    rho=getattr(greeks, "rho", None) if greeks else None,
                )
            )

        return options

    @staticmethod
    def to_dataframe(options: List[OptionsModel]) -> pd.DataFrame:
        return pd.DataFrame([opt.model_dump() for opt in options])
