from zoneinfo import ZoneInfo


EXCHANGE_TIMEZONES = {
    "NMS": ZoneInfo("America/New_York"),
    "NYQ": ZoneInfo("America/New_York"),
    "PCX": ZoneInfo("America/New_York"),
    "CBOE": ZoneInfo("America/Chicago"),
    "LSE": ZoneInfo("Europe/London"),
    "FRA": ZoneInfo("Europe/Berlin"),
}


TENOR_TO_ID = {
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
}

COUPON_FRED = 2
DAY_CONVENTION = "ACT/ACT"
