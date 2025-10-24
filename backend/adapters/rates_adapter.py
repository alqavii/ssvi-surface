import pandas as pd
import math
from fredapi import Fred
import os
from datetime import datetime
from api.data.metadata import TENOR_TO_ID
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FredApiKey = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FredApiKey)


class RatesAdapter:
    @staticmethod
    def generateSOFR():
        dataPart1 = fred.get_series(
            "SOFR", observation_start="2000-01-01", observation_end="2023-01-01"
        )
        dataPart2 = fred.get_series("SOFR", observation_start="2023-01-01")

        dfPart1 = dataPart1.reset_index().rename(columns={"index": "date", 0: "SOFR"})
        dfPart2 = dataPart2.reset_index().rename(columns={"index": "date", 0: "SOFR"})

        rates = (
            pd.concat([dfPart1, dfPart2])
            .drop_duplicates(subset=["date"])
            .sort_values(by="date")
        )
        rates.ffill(inplace=True)

        rates.to_csv(DATA_DIR / "sofr_data.csv", index=False)

    @staticmethod
    def generateTreasuryYields():
        dfs = []
        for tenor, seriesId in TENOR_TO_ID.items():
            dataPart1 = fred.get_series(
                seriesId, observation_start="2000-01-01", observation_end="2023-01-01"
            )
            dataPart2 = fred.get_series(seriesId, observation_start="2023-01-01")

            dfPart1 = dataPart1.reset_index().rename(
                columns={"index": "date", 0: tenor}
            )
            dfPart2 = dataPart2.reset_index().rename(
                columns={"index": "date", 0: tenor}
            )

            df = (
                pd.concat([dfPart1, dfPart2], ignore_index=True)
                .drop_duplicates(subset=["date"])
                .sort_values(by="date")
            )

            df["parYield"] = df[tenor] * 0.01
            df["parYield"] = df["parYield"].round(4)
            df["date"] = pd.to_datetime(df["date"]).dt.date

            dfs.append(df[["date", "parYield"]].rename(columns={"parYield": tenor}))

        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on=["date"], how="outer")
        df = df.set_index("date")

        merged = merged.sort_values("date").drop_duplicates("date")
        merged.ffill(inplace=True)
        merged.to_csv(DATA_DIR / "treasury_par_yields.csv", index=False)

    @staticmethod
    def currentRate(date: datetime):
        if date < datetime.today().date():
            df = pd.read_csv(DATA_DIR / "sofr_data.csv")
            df["date"] = pd.to_datetime(df["date"]).dt.date
            rate = df.loc[df["date"] == date, "SOFR"].values
            if len(rate) > 0:
                return rate[0] * 0.01
            else:
                return None
        if date == datetime.today().date() or date > datetime.today().date():
            df = pd.read_csv(DATA_DIR / "treasury_par_yields.csv")

        return

    @staticmethod
    def getRiskFreeRate(target_date, request_date=None):
        """
        Return the risk-free rate as a decimal for `target_date`.

        - If `target_date` <= `request_date` (defaults to today): use SOFR (last available <= target_date).
        - If `target_date` > `request_date`: use bootstrapped zero rates from the
          treasury par yields observed on `request_date` and log-linear interpolation
          on discount factors.
        """
        from datetime import date as _date
        from api.engines.zero_rates import ZeroRatesEngine
        
        # Normalize dates
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        elif isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        elif isinstance(target_date, _date):
            pass
        else:
            return None

        if request_date is None:
            request_date = datetime.today().date()
        elif isinstance(request_date, datetime):
            request_date = request_date.date()
        elif isinstance(request_date, str):
            request_date = pd.to_datetime(request_date).date()

        # Past or on request date: use SOFR
        if target_date <= request_date:
            try:
                df = pd.read_csv(DATA_DIR / "sofr_data.csv")
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.sort_values("date")
                # pick the latest observation on/before target_date
                df = df[df["date"] <= target_date]
                if df.empty:
                    return None
                sofr_pct = df.iloc[-1]["SOFR"]
                return float(sofr_pct) * 0.01
            except Exception:
                return None

        # Future relative to request date: use zero curve
        try:
            ydf = pd.read_csv(DATA_DIR / "treasury_par_yields.csv")
            ydf["date"] = pd.to_datetime(ydf["date"]).dt.date
            ydf = ydf.sort_values("date")
            # row at or before request_date
            ydf = ydf[ydf["date"] <= request_date]
            if ydf.empty:
                return None
            row = ydf.iloc[-1]

            # Build a tenor -> par yield (decimal) mapping/series
            tenor_cols = [c for c in row.index if c != "date"]
            # Cast tenor column names to float in case they were read as strings
            tenor_map = {float(c): float(row[c]) for c in tenor_cols}
            yields_series = pd.Series(tenor_map).sort_index()

            # Bootstrap zero rates
            zero_list = ZeroRatesEngine.calcZeroRates(yields_series)

            # Map half-year grid to zero rates: 0.5, 1.0, ...
            tenors = [0.5 * (i + 1) for i in range(len(zero_list))]
            zero_curve = pd.Series(zero_list, index=tenors)

            # Year fraction from request_date to target_date (ACT/365.25)
            days = (target_date - request_date).days
            t = max(0.0, days / 365.25)
            if t == 0:
                # immediate horizon; use shortest available zero
                return float(zero_curve.iloc[0])

            # Convert zero rates to discount factors on grid
            df_grid = pd.Series(
                {T: math.exp(-float(r) * T) for T, r in zero_curve.items()}
            ).sort_index()

            # Interpolate ln DF (log-linear on DF), then convert back to zero rate
            grid_T = df_grid.index.to_list()
            grid_lnDF = [math.log(v) for v in df_grid.values]

            # Find bracketing nodes
            if t <= grid_T[0]:
                T1, T2 = grid_T[0], grid_T[1] if len(grid_T) > 1 else grid_T[0]
            elif t >= grid_T[-1]:
                T1, T2 = grid_T[-2], grid_T[-1]
            else:
                # locate interval
                T1 = max(x for x in grid_T if x <= t)
                # next greater
                T2 = min(x for x in grid_T if x >= t and x != T1)

            if T1 == T2:
                lnDF_t = grid_lnDF[grid_T.index(T1)]
            else:
                i1, i2 = grid_T.index(T1), grid_T.index(T2)
                w = (t - T1) / (T2 - T1)
                lnDF_t = grid_lnDF[i1] * (1 - w) + grid_lnDF[i2] * w

            z_t = -lnDF_t / t
            return float(z_t)
        except Exception:
            return None
