from scipy.optimize import brentq
import math
import pandas as pd


class ZeroRatesEngine:
    @staticmethod
    def calcZeroRates(yields: pd.Series) -> list:
        discountRates = {}
        discountRates[0.5] = 100 / (100 + 100 * yields[0.5] / 2)

        for tenor, rate in yields.items():
            if tenor == 0.5:
                pass

            prev = max(discountRates.keys())
            c = 100 * rate / 2
            forwardRate = brentq(
                lambda x: c * (sum(a for a in discountRates.values()))
                + c
                * discountRates[prev]
                * (sum(math.exp(x * -a / 2) for a in range(1, (tenor - prev) * 2)))
                + (100 + c) * math.exp(x * -(tenor - prev))
                - 100,
                -0.1,
                1.0,
            )

            discountRates.update(
                {
                    prev + k * 0.5: discountRates[prev]
                    * math.exp(-forwardRate * (k * 0.5))
                    for k in range(1, ((tenor - prev) * 2) + 1)
                }
            )

        zeroRates = [(-1 / tenor) * math.log(df) for tenor, df in discountRates.items()]

        return zeroRates
