import pandas as pd
import math
from fredapi import Fred
import os
from datetime import datetime, timedelta
from data.metadata import TENOR_TO_ID
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

            dfs.append(
                df[["date", "parYield"]].rename(columns={"parYield": str(tenor)})
            )

        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on=["date"], how="outer")
        df = df.set_index("date")

        merged = merged.sort_values("date").drop_duplicates("date")
        merged.ffill(inplace=True)
        merged.to_csv(DATA_DIR / "treasury_par_yields.csv", index=False)

    @staticmethod
    def getTodayRate():
        return RatesAdapter.getRiskFreeRate(datetime.today().date())

    @staticmethod
    def getRiskFreeRate(target_date, request_date=None):
        from datetime import date as _date
        from engines.zero_rates import ZeroRatesEngine

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

        if target_date <= request_date:
            try:
                df = pd.read_csv(DATA_DIR / "sofr_data.csv")
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.sort_values("date")
                df = df[df["date"] <= target_date]
                if df.empty:
                    return None
                sofr_pct = df.iloc[-1]["SOFR"]
                return float(sofr_pct) * 0.01
            except Exception:
                return None

        try:
            ydf = pd.read_csv(DATA_DIR / "treasury_par_yields.csv")
            ydf["date"] = pd.to_datetime(ydf["date"]).dt.date
            ydf = ydf.sort_values("date")
            ydf = ydf[ydf["date"] <= request_date]
            if ydf.empty:
                return None
            row = ydf.iloc[-1]

            tenor_cols = [c for c in row.index if c != "date"]
            tenor_map = {float(c): float(row[c]) for c in tenor_cols}
            yields_series = pd.Series(tenor_map).sort_index()

            zero_list = ZeroRatesEngine.calcZeroRates(yields_series)

            tenors = [0.5 * (i + 1) for i in range(len(zero_list))]
            zero_curve = pd.Series(zero_list, index=tenors, dtype=float)

            days = (target_date - request_date).days
            t = max(0.0, days / 365.25)
            if t == 0:
                return float(zero_curve.iloc[0])

            df_data = {}
            for T, r in zero_curve.items():
                T_float = float(T)
                r_float = float(r)
                df_data[T_float] = math.exp(-r_float * T_float)
            df_grid = pd.Series(df_data, dtype=float).sort_index()

            grid_T = [float(x) for x in df_grid.index.to_list()]
            grid_lnDF = [math.log(v) for v in df_grid.values]

            if t <= grid_T[0]:
                T1, T2 = grid_T[0], grid_T[1] if len(grid_T) > 1 else grid_T[0]
            elif t >= grid_T[-1]:
                T1, T2 = grid_T[-2], grid_T[-1]
            else:
                T1 = max(x for x in grid_T if x <= t)
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

    @staticmethod
    def updateSOFR():
        try:
            sofr_file = DATA_DIR / "sofr_data.csv"
            if sofr_file.exists():
                existing_df = pd.read_csv(sofr_file)
                existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
                last_date = existing_df["date"].max()
                start_date = last_date + timedelta(days=1)
            else:
                start_date = datetime(2000, 1, 1).date()
                existing_df = pd.DataFrame(columns=["date", "SOFR"])

            new_data = fred.get_series("SOFR", observation_start=start_date)

            if new_data.empty:
                print(f"No new SOFR data available from {start_date}")
                return

            new_df = new_data.reset_index().rename(columns={"index": "date", 0: "SOFR"})
            new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

            new_df = new_df.drop_duplicates(subset=["date"]).sort_values("date")

            new_df["SOFR"] = new_df["SOFR"].ffill()

            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            combined_df = combined_df.drop_duplicates(subset=["date"]).sort_values(
                "date"
            )
            combined_df["SOFR"] = combined_df["SOFR"].ffill()

            combined_df.to_csv(sofr_file, index=False)
            print(f"Updated SOFR data: added {len(new_df)} new records")

        except Exception as e:
            print(f"Error updating SOFR data: {e}")

    @staticmethod
    def updateTreasuryYields():
        try:
            treasury_file = DATA_DIR / "treasury_par_yields.csv"
            if treasury_file.exists():
                existing_df = pd.read_csv(treasury_file)
                existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
                last_date = existing_df["date"].max()
                start_date = last_date + timedelta(days=1)
            else:
                start_date = datetime(2000, 1, 1).date()
                existing_df = pd.DataFrame(columns=["date"])

            new_dfs = []
            for tenor, seriesId in TENOR_TO_ID.items():
                try:
                    new_data = fred.get_series(seriesId, observation_start=start_date)

                    if new_data.empty:
                        continue

                    new_df = new_data.reset_index().rename(
                        columns={"index": "date", 0: tenor}
                    )
                    new_df["parYield"] = new_df[tenor] * 0.01
                    new_df["parYield"] = new_df["parYield"].round(4)
                    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

                    new_df = new_df.drop_duplicates(subset=["date"]).sort_values("date")

                    new_df["parYield"] = new_df["parYield"].ffill()

                    new_df = new_df[["date", "parYield"]].rename(
                        columns={"parYield": str(tenor)}
                    )
                    new_dfs.append(new_df)

                except Exception as e:
                    print(f"Error updating tenor {tenor}: {e}")
                    continue

            if not new_dfs:
                print(f"No new treasury yield data available from {start_date}")
                return

            merged_new = new_dfs[0]
            for df in new_dfs[1:]:
                merged_new = pd.merge(merged_new, df, on=["date"], how="outer")

            merged_new = merged_new.sort_values("date").drop_duplicates("date")
            merged_new = merged_new.ffill()

            if not existing_df.empty and "date" in existing_df.columns:
                combined_df = pd.concat([existing_df, merged_new], ignore_index=True)
            else:
                combined_df = merged_new

            combined_df = combined_df.drop_duplicates(subset=["date"]).sort_values(
                "date"
            )
            combined_df = combined_df.ffill()

            combined_df.to_csv(treasury_file, index=False)
            print(f"Updated treasury yields: added {len(merged_new)} new records")

        except Exception as e:
            print(f"Error updating treasury yields: {e}")

    @staticmethod
    def updateAllRates():
        print("Starting incremental rate updates...")
        RatesAdapter.updateSOFR()
        RatesAdapter.updateTreasuryYields()
        print("Rate updates completed.")

    @staticmethod
    def getLastUpdateDate():
        result = {}

        sofr_file = DATA_DIR / "sofr_data.csv"
        if sofr_file.exists():
            df = pd.read_csv(sofr_file)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            result["sofr"] = df["date"].max()
        else:
            result["sofr"] = None

        treasury_file = DATA_DIR / "treasury_par_yields.csv"
        if treasury_file.exists():
            df = pd.read_csv(treasury_file)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            result["treasury"] = df["date"].max()
        else:
            result["treasury"] = None

        return result
