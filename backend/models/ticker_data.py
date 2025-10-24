from pydantic import BaseModel
from typing import Optional
from zoneinfo import ZoneInfo

class TickerModel(BaseModel):
    spot: float
    dividendYield: float
    exchange: str
    timezone: ZoneInfo

    companyName: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    marketCap: Optional[float] = None
    yearHigh: Optional[float] = None
    yearLow: Optional[float] = None 


