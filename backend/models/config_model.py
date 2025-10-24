from pydantic import BaseModel, Field
from typing import Tuple, Literal
from datetime import datetime, timezone

class Config(BaseModel):
    
    ticker: str
    
    moneyness: Tuple[float, float] = (0.75, 1.25)
    gridPoints: int = 41

    expiryGranularity: Literal["all", "weekly", "monthly"] = "all"
    maxExpiries = 7

    smoothing: Literal["none", "spline"]

    asOf: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
