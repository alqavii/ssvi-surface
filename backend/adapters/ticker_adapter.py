import yfinance as yf
from zoneinfo import ZoneInfo
from models.config_model import Config
from models.ticker_data import TickerModel
from data.metadata import EXCHANGE_TIMEZONES
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import os

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

StockClient = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY
)


class TickerAdapter:
    @staticmethod
    def fetchBasic(cfg: Config) -> TickerModel:
        symbol = StockLatestQuoteRequest(symbol_or_symbols=cfg.ticker)
        latest_quote = StockClient.get_stock_latest_quote(symbol)
        spot = (
            latest_quote[cfg.ticker].ask_price + latest_quote[cfg.ticker].bid_price
        ) / 2
        stock = yf.Ticker(cfg.ticker)
        dividendYield = stock.info.get("dividendYield", 0)
        exchange = stock.info.get("exchange", "N/A")
        timezone = EXCHANGE_TIMEZONES.get(exchange, ZoneInfo("UTC"))
        return TickerModel(
            spot=spot, dividendYield=dividendYield, exchange=exchange, timezone=timezone
        )

    @staticmethod
    def fetchFull(cfg: Config) -> TickerModel:
        base = TickerAdapter.fetchBasic(cfg)
        stock = yf.Ticker(cfg.ticker)
        base.companyName = stock.info.get("longName", "N/A")
        base.sector = stock.info.get("sector", "N/A")
        base.industry = stock.info.get("industry", "N/A")
        base.marketCap = stock.info.get("marketCap", "N/A")
        base.yearHigh = stock.info.get("fiftyTwoWeekHigh", "N/A")
        base.yearLow = stock.info.get("fiftyTwoWeekLow", "N/A")
        return base
