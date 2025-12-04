import numpy as np
import pandas as pd
from scipy.special import ndtr
from models.options_data import OptionType


# Standard normal probability density function (vectorized)
def norm_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)


class IVEngine:
    @staticmethod
    def generateIVSmile(
        options_df: pd.DataFrame,
        rate: float,
        dividendYield: float,
        spot: float,
        optionType: OptionType,
    ) -> pd.DataFrame:
        df = options_df[options_df["optionType"] == optionType.value].copy()
        df = df[["optionType", "strike", "timeToExpiry", "midPrice"]]
        df = df.rename(  # type: ignore
            columns={
                "optionType": "type",
                "strike": "K",
                "timeToExpiry": "T",
                "midPrice": "Price",
            }
        )
        is_call = np.where(df["type"] == OptionType.CALL.value, 1, -1).astype(float)
        df["iv"] = IVEngine._implied_volatility(
            df["Price"], df["K"], df["T"], rate, dividendYield, spot, is_call
        )
        df["S"] = spot  # Add spot price for moneyness calculations

        return df

    @staticmethod
    def _black_scholes(sigma, K, T, r, q, S, is_call):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1 = ndtr(is_call * d1)
        nd2 = ndtr(is_call * d2)

        price = is_call * (S * np.exp(-q * T) * nd1 - K * np.exp(-r * T) * nd2)
        return price

    @staticmethod
    def _vega(sigma, K, T, r, q, S):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)

        return S * np.exp(-q * T) * norm_pdf(d1) * sqrt_T

    @staticmethod
    def _implied_volatility(
        market_price, K, T, r, q, S, is_call, tolerance=1e-5, max_iter=20
    ):
        sigma = np.full_like(market_price, 0.5, dtype=float)

        for _ in range(max_iter):
            price_est = IVEngine._black_scholes(sigma, K, T, r, q, S, is_call)
            vega = IVEngine._vega(sigma, K, T, r, q, S)

            vega = np.where(vega < 1e-8, 1e-8, vega)

            diff = price_est - market_price

            if np.max(np.abs(diff)) < tolerance:
                break

            sigma = sigma - (diff / vega)

            sigma = np.maximum(sigma, 1e-6)

        final_price = IVEngine._black_scholes(sigma, K, T, r, q, S, is_call)
        final_diff = final_price - market_price

        mask_bad = np.abs(final_diff) > tolerance * 10
        sigma[mask_bad] = np.nan

        return sigma

        # ... inside IVEngine class ...

    @staticmethod
    def _ssvi_surface_formula(K, F, theta, phi, rho):
        k = np.log(K / F)
        w = (
            1
            / 2
            * theta
            * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
        )

        return w

        """
        SSVI No-Arbitrage Constraints.
        params: [theta, phi, rho]

        Conditions:
        1. theta > 0
        2. phi > 0
        3. |rho| < 1
        4. theta * phi * (1 + |rho|) < 4  (No Butterfly Arbitrage)
        """
