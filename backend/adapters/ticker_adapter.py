import yfinance as yf
from zoneinfo import ZoneInfo
from api.models.config_model import Config
from api.models.ticker_data import TickerModel
from api.data.metadata import EXCHANGE_TIMEZONES


class TickerAdapter:
    @staticmethod
    def fetchBasic(cfg: Config) -> TickerModel:
        stock = yf.Ticker(cfg.ticker)
        spot = stock.history(period="1d", interval="1m")["Close"].iloc[-1]
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
