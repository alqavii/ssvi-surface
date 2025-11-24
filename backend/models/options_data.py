from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionsRequest(BaseModel):
    ticker: str
    optionType: Optional[OptionType] = None

    expiry: Optional[date] = Field(default=None)
    expiryStart: Optional[date] = Field(default=None)
    expiryEnd: Optional[date] = Field(default=None)

    strike: Optional[float] = Field(default=None)
    strikeMin: Optional[float] = Field(default=None)
    strikeMax: Optional[float] = Field(default=None)

    limit: int = Field(default=250)
