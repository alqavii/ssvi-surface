import os

from adapters.rates_adapter import RatesAdapter


def updateRates():
    fred_key = os.getenv("FRED_API_KEY")
    print(f"FRED_API_KEY present: {'yes' if fred_key else 'no'}")

    print("Starting updateRates...")
    RatesAdapter.updateRates()
    print("updateRates completed.")

    last = RatesAdapter.getLastUpdateDate()
    print(f"Last discount_factors date: {last}")

    print("SOFR update was included.")
