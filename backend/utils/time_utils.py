import pytz
from datetime import datetime
from typing import List
from models.options_data import OptionsModel


def getTimeToExpiry(options: List[OptionsModel]) -> List[OptionsModel]:
    eastern = pytz.timezone("US/Eastern")
    for option in options:
        day, month = option.expiry.day, option.expiry.month
        now = datetime.now(eastern)
        year = now.year
        expiryDate = eastern.localize(datetime(year, month, day, 16, 0, 0))
        option.timeToExpiry = (expiryDate - now).total_seconds() / (365.25 * 24 * 3600)
    return options
