from api.adapters.rates_adapter import RatesAdapter



from fredapi import Fred
import os

FredApiKey = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FredApiKey)

todayRates = fred


#RatesAdapter.generateTreasuryYields()
#RatesAdapter.generateSOFR()




