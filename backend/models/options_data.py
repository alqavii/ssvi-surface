from pydantic import BaseModel
from typing import Optional
from datetime import date
from enum import Enum

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

class OptionsModel(BaseModel):
    ticker: str

    expiry: date
    optionType: OptionType
    strike: float
    lastPrice: float
    bid: float
    ask: float
    midPrice: float

    volume: Optional[int] = None
    openInterest: Optional[int] = None
    lastTradeDate: Optional[date] = None

    impliedVol: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None

