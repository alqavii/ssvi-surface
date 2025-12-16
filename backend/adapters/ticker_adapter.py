import yfinance as yf
from zoneinfo import ZoneInfo
from models.ticker_data import TickerModel
from data.metadata import EXCHANGE_TIMEZONES
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import os

ALPACA_API_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

StockClient = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY
)


class TickerAdapter:
    @staticmethod
    def fetchBasic(ticker: str) -> TickerModel:
        symbol = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        latest_quote = StockClient.get_stock_latest_quote(symbol)
        spot = (latest_quote[ticker].ask_price + latest_quote[ticker].bid_price) / 2
        stock = yf.Ticker(ticker)
        yahoo_spot = stock.history(period="1d")["Close"].iloc[-1]
        if abs(spot - yahoo_spot) / yahoo_spot > 0.01:
            spot = yahoo_spot  # Use Yahoo spot if Alpaca quote is stale
        dividendYield = stock.info.get("dividendYield", 0)
        exchange = stock.info.get("exchange", "N/A")
        timezone = EXCHANGE_TIMEZONES.get(exchange, ZoneInfo("UTC"))
        return TickerModel(
            spot=spot, dividendYield=dividendYield, exchange=exchange, timezone=timezone
        )

    @staticmethod
    def fetchFull(ticker: str) -> TickerModel:
        base = TickerAdapter.fetchBasic(ticker)
        stock = yf.Ticker(ticker)
        base.companyName = stock.info.get("longName", "N/A")
        base.sector = stock.info.get("sector", "N/A")
        base.industry = stock.info.get("industry", "N/A")
        base.marketCap = stock.info.get("marketCap", "N/A")
        base.yearHigh = stock.info.get("fiftyTwoWeekHigh", "N/A")
        base.yearLow = stock.info.get("fiftyTwoWeekLow", "N/A")
        return base
