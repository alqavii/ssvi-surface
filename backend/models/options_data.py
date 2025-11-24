from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionsRequest(BaseModel):
    """
    Minimal request object for fetching options via Alpaca.
    All filters map directly to OptionChainRequest parameters.
    """

    ticker: str
    optionType: Optional[OptionType] = None

    # Expiry filters
    expiry: Optional[date] = Field(default=None, description="Exact expiry date")
    expiryStart: Optional[date] = Field(
        default=None, description="Earliest expiry date (inclusive)"
    )
    expiryEnd: Optional[date] = Field(
        default=None, description="Latest expiry date (inclusive)"
    )

    # Strike filters
    strike: Optional[float] = Field(
        default=None, description="Exact strike price to match"
    )
    strikeMin: Optional[float] = Field(
        default=None, description="Minimum strike price (inclusive)"
    )
    strikeMax: Optional[float] = Field(
        default=None, description="Maximum strike price (inclusive)"
    )

    limit: int = Field(
        default=250, description="Maximum number of contracts to return (post-fetch)"
    )


class OptionsModel(BaseModel):
    ticker: str

    expiry: date
    timeToExpiry: Optional[float] = None
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
