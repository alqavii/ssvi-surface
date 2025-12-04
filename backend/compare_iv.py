import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
from math import isfinite
from engines.IV_smile import IVEngine


class IVCompare:
    def __init__(self, rate=0.05, dividend=0.01, spot=100.0):
        self.rate = rate
        self.dividend = dividend
        self.spot = spot

    def _blackScholesCall(self, sigma, K, T):
        r = self.rate
        q = self.dividend
        S = self.spot
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

    def impliedVolatility_ref(self, K, T, C):
        def fsafe(sigma):
            pricediff = self._blackScholesCall(sigma, K, T) - C
            if not isfinite(pricediff):
                raise ValueError("f returned bad value")
            if pricediff > abs(C) * 50:
                raise ValueError("f returned too large value")
            return pricediff

        f = lambda sigma: self._blackScholesCall(sigma, K, T) - C

        try:
            return root_scalar(
                f, bracket=[1e-6, 5.0], method="brentq", x0=0.05, x1=0.5
            ).root
        except:
            try:
                return root_scalar(
                    f, bracket=[1e-2, 5.0], method="brentq", x0=0.2, x1=0.4
                ).root
            except:
                return np.nan

    def compare(self):
        # Generate synthetic data
        strikes = np.linspace(80, 120, 41)
        T = 0.5

        # Generate theoretical prices using a known volatility (e.g., 0.3)
        known_vol = 0.3
        prices = [self._blackScholesCall(known_vol, k, T) for k in strikes]
        prices_array = np.array(prices)

        print(
            f"Comparing IV Calculation Methods (Spot={self.spot}, T={T}, True Vol={known_vol})"
        )
        print("-" * 80)
        print(
            f"{'Strike':<10} | {'Price':<10} | {'Ref (Brent)':<12} | {'Engine (Newton)':<15} | {'Diff':<10}"
        )
        print("-" * 80)

        # Calculate using Reference Logic
        ref_ivs = []
        for k, p in zip(strikes, prices):
            ref_ivs.append(self.impliedVolatility_ref(k, T, p))

        # Calculate using Engine Logic
        times = np.full_like(strikes, T)
        is_call = np.ones_like(strikes)

        engine_ivs = IVEngine._vectorized_implied_volatility(
            prices_array, strikes, times, self.rate, self.dividend, self.spot, is_call
        )

        errors = []
        for i, k in enumerate(strikes):
            ref = ref_ivs[i]
            eng = engine_ivs[i]
            diff = abs(ref - eng)
            errors.append(diff)

            if i % 5 == 0:  # Print every 5th row to keep output manageable
                print(
                    f"{k:<10.1f} | {prices[i]:<10.4f} | {ref:<12.6f} | {eng:<15.6f} | {diff:.2e}"
                )

        avg_error = np.mean(errors)
        max_error = np.max(errors)

        print("-" * 80)
        print(f"Average Diff: {avg_error:.2e}")
        print(f"Max Diff:     {max_error:.2e}")

        if avg_error < 1e-5:
            print("\nPASSED: Both engines produce virtually identical results.")
        else:
            print("\nWARNING: Significant divergence detected between solvers.")


if __name__ == "__main__":
    comp = IVCompare()
    comp.compare()
