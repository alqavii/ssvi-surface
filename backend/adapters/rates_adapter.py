import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred
from data.metadata import TENOR_TO_ID, TBILLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
fred = Fred(api_key=os.getenv("FRED_API_KEY"))


class RatesAdapter:
    """Rates adapter focused on discount factors and SOFR updates.

    Public methods:
    - generateZeroCurves
    - updateZeroCurves
    - updateRates
    - updateSOFR
    - getLastUpdateDate
    """

    _updating = False

    # --- Internal utilities (condensed) ---
    @staticmethod
    def _csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    @staticmethod
    def _fred(series_id: str, start_date) -> pd.DataFrame:
        try:
            data = fred.get_series(series_id, observation_start=start_date)
            if data.empty:
                return pd.DataFrame()
            df = data.reset_index().rename(columns={"index": "date", 0: "value"})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df.drop_duplicates("date").sort_values("date")
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _zero_curve_on_or_before(target_date):
        df = RatesAdapter._csv(DATA_DIR / "discount_factors.csv")
        if df.empty:
            return None
        df = df[df["date"] <= target_date]
        if df.empty:
            return None
        row = df.iloc[-1]
        curve = {float(c): float(row[c]) for c in row.index if c != "date"}
        return pd.Series(curve).sort_index()

    @staticmethod
    def _interp_zero_rate(curve: pd.Series, t):
        """Deprecated: Interpolation removed; keeping stub for compatibility."""
        raise NotImplementedError("Zero rate interpolation is not used")

    @staticmethod
    def generateZeroCurves():
        """Bootstrap full discount factors from 2000-01-01 using FRED data."""
        from engines.zero_rates import ZeroRatesEngine

        # Par yields
        par_dfs = []
        for tenor, sid in TENOR_TO_ID.items():
            df = (
                pd.concat(
                    [
                        RatesAdapter._fred(sid, "2000-01-01")[
                            lambda x: x["date"] < datetime(2023, 1, 1).date()
                        ],
                        RatesAdapter._fred(sid, "2023-01-01"),
                    ]
                )
                .drop_duplicates("date")
                .sort_values("date")
            )
            df[str(tenor)] = (df["value"] * 0.01).round(4)
            par_dfs.append(df[["date", str(tenor)]])

        par = par_dfs[0]
        for df in par_dfs[1:]:
            par = pd.merge(par, df, on="date", how="outer")
        par = par.sort_values("date").drop_duplicates("date").ffill()

        # T-bills
        tbill_dfs = []
        for tenor, sid in TBILLS.items():
            df = (
                pd.concat(
                    [
                        RatesAdapter._fred(sid, "2000-01-01")[
                            lambda x: x["date"] < datetime(2023, 1, 1).date()
                        ],
                        RatesAdapter._fred(sid, "2023-01-01"),
                    ]
                )
                .drop_duplicates("date")
                .sort_values("date")
            )
            df[str(tenor)] = (df["value"] * 0.01).round(4)
            tbill_dfs.append(df[["date", str(tenor)]])

        tbill = tbill_dfs[0]
        for df in tbill_dfs[1:]:
            tbill = pd.merge(tbill, df, on="date", how="outer")
        tbill = tbill.sort_values("date").drop_duplicates("date").ffill()

        # Build discount factors
        dates = sorted(set(par["date"]) | set(tbill["date"]))
        rows = []
        for d in dates:
            prow = par[par["date"] == d]
            if prow.empty:
                continue
            # Skip if any par yield is missing (avoids engine root errors)
            if pd.isna(prow.drop(columns=["date"]).iloc[0]).any():
                continue
            pseries = pd.Series(
                {float(c): float(prow.iloc[0][c]) for c in prow.columns if c != "date"}
            ).sort_index()
            dfs = ZeroRatesEngine.calcZeroRates(pseries)
            trow = tbill[tbill["date"] == d]
            if not trow.empty:
                tr = trow.iloc[0]
                for tenor in TBILLS.keys():
                    col = str(tenor)
                    if col in tr and not pd.isna(tr[col]):
                        dfs[tenor] = 1.0 / (1.0 + float(tr[col]) * tenor)
            # Only store tracked bill and bond tenors; include all columns every row
            desired = sorted(list(TBILLS.keys()) + list(TENOR_TO_ID.keys()))
            row_dict = {"date": d}
            for t in desired:
                row_dict[str(t)] = dfs.get(t, np.nan)
            rows.append(row_dict)

        pd.DataFrame(rows).sort_values("date").drop_duplicates("date").to_csv(
            DATA_DIR / "discount_factors.csv", index=False
        )

    @staticmethod
    def updateSOFR():
        """Update SOFR daily series into sofr_data.csv from FRED (series 'SOFR')."""
        try:
            file_path = DATA_DIR / "sofr_data.csv"
            existing = RatesAdapter._csv(file_path)
            last_date = existing["date"].max() if not existing.empty else None
            start_date = (
                (last_date + timedelta(days=1))
                if last_date
                else datetime(2018, 4, 1).date()
            )
            df = RatesAdapter._fred("SOFR", start_date)
            if df.empty:
                print(f"No new SOFR data available from {start_date}")
                return
            df = df.rename(columns={"value": "SOFR"})
            combined = (
                (
                    pd.concat([existing, df], ignore_index=True)
                    if not existing.empty
                    else df
                )
                .drop_duplicates("date")
                .sort_values("date")
            )
            combined.to_csv(file_path, index=False)
            print(f"Updated SOFR: added {len(df)} new records")
        except Exception as e:
            print(f"Error updating SOFR: {e}")

    @staticmethod
    # Removed: getRiskFreeRate and Today proxies

    @staticmethod
    def updateZeroCurves():
        """Append latest discount factors to `discount_factors.csv` from last date."""
        from engines.zero_rates import ZeroRatesEngine

        try:
            file_path = DATA_DIR / "discount_factors.csv"
            existing = RatesAdapter._csv(file_path)
            last_date = existing["date"].max() if not existing.empty else None
            start_date = (
                (last_date + timedelta(days=1))
                if last_date
                else datetime(2000, 1, 1).date()
            )

            # Par yields
            par_dfs = []
            for tenor, sid in TENOR_TO_ID.items():
                df = RatesAdapter._fred(sid, start_date)
                if df.empty:
                    continue
                df[str(tenor)] = (df["value"] * 0.01).round(4)
                par_dfs.append(df[["date", str(tenor)]].ffill())
            if not par_dfs:
                print(f"No new discount factor data available from {start_date}")
                return
            par = par_dfs[0]
            for df in par_dfs[1:]:
                par = pd.merge(par, df, on="date", how="outer")
            par = par.sort_values("date").drop_duplicates("date").ffill()

            # T-bills (optional)
            tbill = None
            tbill_dfs = []
            for tenor, sid in TBILLS.items():
                df = RatesAdapter._fred(sid, start_date)
                if df.empty:
                    continue
                df[str(tenor)] = (df["value"] * 0.01).round(4)
                tbill_dfs.append(df[["date", str(tenor)]].ffill())
            if tbill_dfs:
                tbill = tbill_dfs[0]
                for df in tbill_dfs[1:]:
                    tbill = pd.merge(tbill, df, on="date", how="outer")
                tbill = tbill.sort_values("date").drop_duplicates("date").ffill()

            # Build rows
            new_rows = []
            for _, prow in par.iterrows():
                d = prow["date"]
                # Skip rows with missing par yields
                if pd.isna(
                    pd.Series({c: prow[c] for c in par.columns if c != "date"})
                ).any():
                    continue
                pseries = pd.Series(
                    {float(c): float(prow[c]) for c in par.columns if c != "date"}
                ).sort_index()
                dfs = ZeroRatesEngine.calcZeroRates(pseries)
                if tbill is not None:
                    trow = tbill[tbill["date"] == d]
                    if not trow.empty:
                        tr = trow.iloc[0]
                        for tenor in TBILLS.keys():
                            col = str(tenor)
                            if col in tr and not pd.isna(tr[col]):
                                dfs[tenor] = 1.0 / (1.0 + float(tr[col]) * tenor)
                # Only store tracked bill and bond tenors; include all columns every row
                desired = sorted(list(TBILLS.keys()) + list(TENOR_TO_ID.keys()))
                row_dict = {"date": d}
                for t in desired:
                    row_dict[str(t)] = dfs.get(t, np.nan)
                new_rows.append(row_dict)

            new_df = pd.DataFrame(new_rows).sort_values("date").drop_duplicates("date")
            combined = (
                (
                    pd.concat([existing, new_df], ignore_index=True)
                    if not existing.empty
                    else new_df
                )
                .drop_duplicates("date")
                .sort_values("date")
            )
            combined.to_csv(file_path, index=False)
            print(f"Updated discount factors: added {len(new_df)} new records")
        except Exception as e:
            print(f"Error updating discount factors: {e}")

    @staticmethod
    def _is_data_stale():
        yesterday = datetime.today().date() - timedelta(days=1)
        df = RatesAdapter._csv(DATA_DIR / "discount_factors.csv")
        last = df["date"].max() if not df.empty else None
        return last is None or last < yesterday

    @staticmethod
    def _ensure_data_updated(silent=False, skip_if_updating=False):
        if skip_if_updating and RatesAdapter._updating:
            return
        if RatesAdapter._is_data_stale():
            if not silent:
                print("Discount factor data is stale, updating...")
            RatesAdapter._updating = True
            try:
                RatesAdapter.updateRates()
            finally:
                RatesAdapter._updating = False

    @staticmethod
    def updateRates():
        """Update discount factors and SOFR."""
        print("Starting rate updates...")
        RatesAdapter.updateZeroCurves()
        RatesAdapter.updateSOFR()
        print("Rate updates completed.")

    @staticmethod
    # Removed: updateRiskFreeRates (no longer maintained)

    @staticmethod
    def getLastUpdateDate():
        df = RatesAdapter._csv(DATA_DIR / "discount_factors.csv")
        return df["date"].max() if not df.empty else None
