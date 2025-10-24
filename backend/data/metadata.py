from zoneinfo import ZoneInfo

from pydantic import BaseModel

EXCHANGE_TIMEZONES = {
    "NMS": ZoneInfo("America/New_York"),   # NASDAQ
    "NYQ": ZoneInfo("America/New_York"),   # NYSE
    "PCX": ZoneInfo("America/New_York"),   # ARCA
    "CBOE": ZoneInfo("America/Chicago"),   # CBOE
    "LSE": ZoneInfo("Europe/London"),      # London
    "FRA": ZoneInfo("Europe/Berlin"),      # Frankfurt
}


TENOR_TO_ID = {
    0.5:  "DGS6MO",   
    1.0:   "DGS1",     
    2.0:   "DGS2",     
    3.0:   "DGS3",     
    5.0:   "DGS5",     
    7.0:   "DGS7",     
    10.0:  "DGS10",  
}

COUPON_FRED = 2
DAY_CONVENTION = "ACT/ACT"
