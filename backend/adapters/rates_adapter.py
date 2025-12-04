import pandas as pd
import math
from fredapi import Fred
import os
from datetime import datetime, timedelta
from data.metadata import TENOR_TO_ID
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
fred = Fred(api_key=os.getenv("FRED_API_KEY"))


class RatesAdapter:
    _updating = False

    @staticmethod
    def _normalize_date(date_val):
        """Convert various date formats to date object."""
        if isinstance(date_val, datetime):
            return date_val.date()
        if isinstance(date_val, str):
            return pd.to_datetime(date_val).date()
        return date_val if hasattr(date_val, "year") else None

    @staticmethod
    def _read_csv_with_dates(file_path):
        """Read CSV and convert date column."""
        if not file_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    @staticmethod
    def _get_last_date(file_path):
        """Get the last date from a CSV file."""
        df = RatesAdapter._read_csv_with_dates(file_path)
        return df["date"].max() if not df.empty else None

    @staticmethod
    def _fetch_fred_series(series_id, start_date):
        """Fetch FRED series and normalize to DataFrame."""
        try:
            data = fred.get_series(series_id, observation_start=start_date)
            if data.empty:
                return pd.DataFrame()
            df = data.reset_index().rename(columns={"index": "date", 0: "value"})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df.drop_duplicates(subset=["date"]).sort_values("date")
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _merge_and_save(df, file_path, value_col="value"):
        """Merge new data with existing, deduplicate, and save."""
        existing = RatesAdapter._read_csv_with_dates(file_path)
        combined = (
            pd.concat([existing, df], ignore_index=True) if not existing.empty else df
        )
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
        if value_col in combined.columns:
            combined[value_col] = combined[value_col].ffill()
        combined.to_csv(file_path, index=False)
        return len(df)

    @staticmethod
    def _get_zero_curve_for_date(request_date):
        """Get zero curve for a specific date from Treasury par yields."""
        from engines.zero_rates import ZeroRatesEngine

        df = RatesAdapter._read_csv_with_dates(DATA_DIR / "zero_curves.csv")
        df = df[df["date"] <= request_date]
        if df.empty:
            return None

        row = df.iloc[-1]
        tenor_map = {float(c): float(row[c]) for c in row.index if c != "date"}
        yields_series = pd.Series(tenor_map).sort_index()
        zero_list = ZeroRatesEngine.calcZeroRates(yields_series)
        return pd.Series(
            zero_list, index=[0.5 * (i + 1) for i in range(len(zero_list))]
        )

    @staticmethod
    def _interpolate_zero_rate(zero_curve, t):
        """Interpolate zero rate for time t from zero curve."""
        if t == 0:
            return float(zero_curve.iloc[0])

        df_grid = pd.Series(
            {T: math.exp(-r * T) for T, r in zero_curve.items()}
        ).sort_index()
        grid_T, grid_lnDF = (
            list(df_grid.index),
            [math.log(v) for v in df_grid.values],
        )

        if t <= grid_T[0]:
            T1, T2 = grid_T[0], grid_T[1] if len(grid_T) > 1 else grid_T[0]
        elif t >= grid_T[-1]:
            T1, T2 = grid_T[-2], grid_T[-1]
        else:
            T1 = max(x for x in grid_T if x <= t)
            T2 = min(x for x in grid_T if x >= t and x != T1)

        i1, i2 = grid_T.index(T1), grid_T.index(T2)
        w = (t - T1) / (T2 - T1) if T1 != T2 else 0
        lnDF_t = grid_lnDF[i1] * (1 - w) + grid_lnDF[i2] * w
        return float(-lnDF_t / t)

    @staticmethod
    def generateZeroCurves():
        """Generate initial zero curve data from FRED Treasury yields."""
        dfs = []
        for tenor, series_id in TENOR_TO_ID.items():
            df1 = RatesAdapter._fetch_fred_series(series_id, "2000-01-01")
            df1 = df1[df1["date"] < datetime(2023, 1, 1).date()]
            df2 = RatesAdapter._fetch_fred_series(series_id, "2023-01-01")
            df = (
                pd.concat([df1, df2])
                .drop_duplicates(subset=["date"])
                .sort_values("date")
            )
            df["parYield"] = (df["value"] * 0.01).round(4)
            df = df[["date", "parYield"]].rename(columns={"parYield": str(tenor)})
            dfs.append(df)

        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="date", how="outer")
        merged = merged.sort_values("date").drop_duplicates("date").ffill()
        merged.to_csv(DATA_DIR / "zero_curves.csv", index=False)

    @staticmethod
    def getTodayRate():
        """Get today's risk-free rate using zero curve (uses yesterday's rate)."""
        RatesAdapter._ensure_data_updated(silent=True)
        yesterday = datetime.today().date() - timedelta(days=1)
        return RatesAdapter.getRiskFreeRate(yesterday, yesterday)

    @staticmethod
    def getRiskFreeRate(target_date, request_date=None, skip_update_check=False):
        """Get risk-free rate for a target date using Treasury zero curve."""
        if not skip_update_check:
            RatesAdapter._ensure_data_updated(silent=True, skip_if_updating=True)

        target_date = RatesAdapter._normalize_date(target_date)
        request_date = (
            RatesAdapter._normalize_date(request_date) or datetime.today().date()
        )
        if not target_date:
            return None

        try:
            zero_curve = RatesAdapter._get_zero_curve_for_date(request_date)
            if zero_curve is None:
                return None

            t = max(0.0, (target_date - request_date).days / 365.25)
            return RatesAdapter._interpolate_zero_rate(zero_curve, t)
        except Exception:
            return None

    @staticmethod
    def updateZeroCurves():
        """Update zero curve data from FRED API."""
        try:
            file_path = DATA_DIR / "zero_curves.csv"
            last_date = RatesAdapter._get_last_date(file_path)
            start_date = (
                (last_date + timedelta(days=1))
                if last_date
                else datetime(2000, 1, 1).date()
            )

            new_dfs = []
            for tenor, series_id in TENOR_TO_ID.items():
                try:
                    df = RatesAdapter._fetch_fred_series(series_id, start_date)
                    if df.empty:
                        continue
                    df["parYield"] = (df["value"] * 0.01).round(4)
                    df = df[["date", "parYield"]].rename(
                        columns={"parYield": str(tenor)}
                    )
                    df[str(tenor)] = df[str(tenor)].ffill()
                    new_dfs.append(df)
                except Exception as e:
                    print(f"Error updating tenor {tenor}: {e}")

            if not new_dfs:
                print(f"No new zero curve data available from {start_date}")
                return

            merged = new_dfs[0]
            for df in new_dfs[1:]:
                merged = pd.merge(merged, df, on="date", how="outer")
            merged = merged.sort_values("date").drop_duplicates("date").ffill()

            existing = RatesAdapter._read_csv_with_dates(file_path)
            combined = (
                pd.concat([existing, merged], ignore_index=True)
                if not existing.empty
                else merged
            )
            combined = (
                combined.drop_duplicates(subset=["date"]).sort_values("date").ffill()
            )
            combined.to_csv(file_path, index=False)
            print(f"Updated zero curves: added {len(merged)} new records")
        except Exception as e:
            print(f"Error updating zero curves: {e}")

    @staticmethod
    def _is_data_stale():
        """Check if zero curve data is stale (older than yesterday)."""
        yesterday = datetime.today().date() - timedelta(days=1)
        last_update = RatesAdapter._get_last_date(DATA_DIR / "zero_curves.csv")
        return last_update is None or last_update < yesterday

    @staticmethod
    def _ensure_data_updated(silent=False, skip_if_updating=False):
        """Ensure zero curve data is up to date. Updates if stale."""
        if skip_if_updating and RatesAdapter._updating:
            return
        if RatesAdapter._is_data_stale():
            if not silent:
                print("Zero curve data is stale, updating...")
            RatesAdapter._updating = True
            try:
                RatesAdapter.updateRates(include_risk_free=True)
            finally:
                RatesAdapter._updating = False

    @staticmethod
    def updateRates(include_risk_free=True):
        """Update zero curves and optionally risk-free rates."""
        print("Starting rate updates...")
        RatesAdapter.updateZeroCurves()
        if include_risk_free:
            RatesAdapter.updateRiskFreeRates()
        print("Rate updates completed.")

    @staticmethod
    def updateRiskFreeRates():
        """Update risk_free.csv using zero curves."""
        try:
            file_path = DATA_DIR / "risk_free.csv"
            yesterday = datetime.today().date() - timedelta(days=1)
            last_date = RatesAdapter._get_last_date(file_path)
            start_date = (
                (last_date + timedelta(days=1))
                if last_date
                else datetime(2000, 1, 1).date()
            )

            if start_date > yesterday:
                print(
                    f"Risk-free rates are already up to date (last date: {last_date})"
                )
                return

            # Get all dates from zero curves
            zero_curve_df = RatesAdapter._read_csv_with_dates(
                DATA_DIR / "zero_curves.csv"
            )
            date_range = zero_curve_df[
                (zero_curve_df["date"] >= start_date)
                & (zero_curve_df["date"] <= yesterday)
            ]["date"]

            if date_range.empty:
                print("No new dates to process")
                return

            # Calculate rates for each date (using zero curve for t=0, i.e., spot rate)
            new_rates = []
            for d in date_range:
                rate = RatesAdapter.getRiskFreeRate(d, d, skip_update_check=True)
                if rate is not None:
                    new_rates.append({"date": d, "value": rate * 100})

            if not new_rates:
                print("No new risk-free rates to add")
                return

            new_df = pd.DataFrame(new_rates)
            new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
            count = RatesAdapter._merge_and_save(new_df, file_path)
            print(
                f"Updated risk-free rates: added {count} new records (from zero curves)"
            )
        except Exception as e:
            print(f"Error updating risk-free rates: {e}")
            import traceback

            traceback.print_exc()

    @staticmethod
    def getLastUpdateDate():
        """Return last update date for zero curve data."""
        return RatesAdapter._get_last_date(DATA_DIR / "zero_curves.csv")
