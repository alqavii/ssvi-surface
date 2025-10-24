import yfinance as yf
import pandas as pd
from typing import Dict, List, cast, Any
from models.config_model import Config
from models.ticker_data import TickerModel
from models.options_data import OptionsModel, OptionType
from time import sleep

class OptionsAdapter:
    @staticmethod
    def fetchOptions(cfg: Config, ticker: TickerModel) -> List[OptionsModel]:
        stock = yf.Ticker(cfg.ticker)
        spot = ticker.spot
        allOptions = []
        for i in range(0, cfg.maxExpiries):
            sleep(0.2)
            temp = stock.option_chain(stock.options[i])
            expiry = stock.options[i]
            calls = temp.calls[["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "lastTradeDate"]]
            calls["optionType"] = OptionType.CALL
            puts = temp.puts[["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "lastTradeDate"]]
            puts["optionType"] = OptionType.PUT
            prices = pd.concat([calls, puts], ignore_index=True).sort_values(by="strike")
            prices = prices[(prices["strike"] <= cfg.moneyness[1]*spot) & (prices["strike"] >= spot*cfg.moneyness[0])].sort_values(by='strike')
            prices["expiry"] = expiry
            prices["midPrice"] = (prices["bid"] + prices["ask"])/2
            prices = prices[["expiry", "optionType", "strike", "lastPrice", "bid", "ask", "midPrice", "volume", "openInterest", "lastTradeDate"]]
            
            records = cast(List[Dict[str, Any]], prices.to_dict(orient="records"))

            allOptions.extend(
                [
                OptionsModel(
                    ticker=cfg.ticker,
                    **row,
                    impliedVol = None,
                    delta = None,
                    gamma = None,
                    vega = None,
                    theta = None,
                    rho = None,
                )
                for row in records
                ]
            )
        return allOptions