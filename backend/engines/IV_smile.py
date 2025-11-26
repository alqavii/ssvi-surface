import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.interpolate import griddata, UnivariateSpline
from scipy.ndimage import gaussian_filter


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
    ) -> pd.DataFrame:
        if options_df.empty:
            options_df["iv"] = []
            options_df["smoothed_iv"] = []
            return options_df

        required_cols = ["strike", "timeToExpiry", "midPrice", "optionType"]
        if not all(col in options_df.columns for col in required_cols):
            options_df["iv"] = np.nan
            return options_df

        strikes = options_df["strike"].values
        times = options_df["timeToExpiry"].values
        prices = options_df["midPrice"].values

        is_call = np.where(
            options_df["optionType"].astype(str).str.upper() == "CALL", 1.0, -1.0
        )

        # 1. Calculate Raw IVs (Vectorized Newton-Raphson)
        iv_array = IVEngine._vectorized_implied_volatility(
            prices, strikes, times, rate, dividendYield, spot, is_call
        )
        options_df["iv"] = iv_array

        # 2. Apply Per-Expiry Spline Smoothing (mimicking reference.py)
        # This ensures each slice is a smooth curve before we even try 3D interpolation
        options_df["smoothed_iv"] = np.nan

        # We iterate over unique expiries (much faster than iterating over rows)
        for expiry in options_df["expiry"].unique():
            mask = options_df["expiry"] == expiry
            expiry_df = options_df[mask]

            # Filter for valid IVs for this slice
            valid_slice = expiry_df.dropna(subset=["iv", "strike"])
            valid_slice = valid_slice[valid_slice["iv"] > 0]

            if len(valid_slice) > 3:
                # Sort by strike for spline
                valid_slice = valid_slice.sort_values("strike")
                x = valid_slice["strike"].values
                y = valid_slice["iv"].values

                try:
                    # s=0.05 provides a balance between fitting points and smoothing noise
                    spline = UnivariateSpline(x, y, k=3, s=0.05)

                    # Predict smoothed IV for ALL strikes in this expiry (even those with missing raw IV)
                    # We clamp the output to be positive
                    smoothed = np.maximum(0.01, spline(expiry_df["strike"].values))
                    options_df.loc[mask, "smoothed_iv"] = smoothed
                except Exception:
                    # Fallback to raw IV if spline fails
                    options_df.loc[mask, "smoothed_iv"] = options_df.loc[mask, "iv"]
            else:
                # Not enough points to smooth, use raw
                options_df.loc[mask, "smoothed_iv"] = options_df.loc[mask, "iv"]

        return options_df

    @staticmethod
    def generate_surface_mesh(df: pd.DataFrame, grid_size: int = 60) -> dict:
        """
        Generates a 3D mesh using the 'smoothed_iv' column for high-quality visualization.
        Matches the reference.py logic of cubic interpolation + spline pre-smoothing.
        """
        if df.empty or "smoothed_iv" not in df.columns:
            return {}

        # Use the smoothed column if available, else raw iv
        target_col = "smoothed_iv" if "smoothed_iv" in df.columns else "iv"

        valid_data = df.dropna(subset=["strike", "timeToExpiry", target_col])
        valid_data = valid_data[valid_data[target_col] > 0]

        if valid_data.empty:
            return {}

        x = valid_data["strike"].values
        y = valid_data["timeToExpiry"].values
        z = valid_data[target_col].values

        # Create a regular grid
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        Xi, Yi = np.meshgrid(xi, yi)

        # Cubic interpolation on the already-smoothed data
        Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

        # Fill holes with nearest neighbor (reference style robust filling)
        mask = np.isnan(Zi)
        if np.any(mask):
            Zi_nearest = griddata((x, y), z, (Xi, Yi), method="nearest")
            Zi[mask] = Zi_nearest[mask]

        # Final light touch Gaussian smoothing for the 3D transition between expiries
        Zi = gaussian_filter(Zi, sigma=0.5)

        return {"x": xi.tolist(), "y": yi.tolist(), "z": Zi.tolist()}

    @staticmethod
    def _vectorized_black_scholes(sigma, K, T, r, q, S, is_call):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1 = ndtr(is_call * d1)
        nd2 = ndtr(is_call * d2)

        price = is_call * (S * np.exp(-q * T) * nd1 - K * np.exp(-r * T) * nd2)
        return price

    @staticmethod
    def _vectorized_vega(sigma, K, T, r, q, S):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)

        return S * np.exp(-q * T) * norm_pdf(d1) * sqrt_T

    @staticmethod
    def _vectorized_implied_volatility(
        market_price, K, T, r, q, S, is_call, tolerance=1e-5, max_iter=20
    ):
        sigma = np.full_like(market_price, 0.5, dtype=float)

        for _ in range(max_iter):
            price_est = IVEngine._vectorized_black_scholes(
                sigma, K, T, r, q, S, is_call
            )
            vega = IVEngine._vectorized_vega(sigma, K, T, r, q, S)

            vega = np.where(vega < 1e-8, 1e-8, vega)

            diff = price_est - market_price

            if np.max(np.abs(diff)) < tolerance:
                break

            sigma = sigma - (diff / vega)

            sigma = np.maximum(sigma, 1e-6)

        final_price = IVEngine._vectorized_black_scholes(sigma, K, T, r, q, S, is_call)
        final_diff = final_price - market_price

        mask_bad = np.abs(final_diff) > tolerance * 10
        sigma[mask_bad] = np.nan

        return sigma
