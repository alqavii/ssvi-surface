from models.config_model import IVConfig
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar
from math import isfinite


class IVEngine:
    @staticmethod
    def generateIVSmile(
        cfg: IVConfig,
        options_df: pd.DataFrame,
        rate: float,
        dividendYield: float,
        spot: float,
    ):
        if options_df.empty:
            return []

        required_cols = ["strike", "timeToExpiry", "midPrice"]
        if not all(col in options_df.columns for col in required_cols):
            return []

        ivs = []
        for _, row in options_df.iterrows():
            iv = IVEngine._impliedVolatility(
                row["strike"],
                row["timeToExpiry"],
                row["midPrice"],
                rate,
                dividendYield,
                spot,
            )
            ivs.append(iv)
        return ivs

    @staticmethod
    def _blackScholesCall(
        sigma: float, K: float, T: float, r: float, q: float, S: float
    ) -> float:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

    @staticmethod
    def _impliedVolatility(
        K: float, T: float, C: float, r: float, q: float, S: float
    ) -> float:
        def f(sigma: float) -> float:
            pricediff = IVEngine._blackScholesCall(sigma, K, T, r, q, S) - C
            if not isfinite(pricediff):
                raise ValueError(
                    f"f returned bad value for sigma={sigma} and K={K}, T={T}, C={C}"
                )
            if pricediff > abs(C) * 50:
                raise ValueError(
                    f"f returned too large value for sigma={sigma} and K={K}, T={T}, C={C}"
                )
            return pricediff

        try:
            return root_scalar(
                f, bracket=[1e-6, 5.0], method="brentq", x0=0.05, x1=0.5
            ).root
        except (ValueError, RuntimeError):
            try:
                return root_scalar(
                    f, bracket=[1e-2, 5.0], method="brentq", x0=0.2, x1=0.4
                ).root
            except (ValueError, RuntimeError):
                return np.nan
