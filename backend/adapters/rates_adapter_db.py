import pandas as pd
import math
from fredapi import Fred
import os
from datetime import datetime, timedelta
from data.metadata import TENOR_TO_ID
from pathlib import Path
from database.services import DatabaseRatesService
from database.config import SessionLocal
from database.models import SOFRRate, TreasuryYield

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FredApiKey = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FredApiKey)


class RatesAdapter:
    @staticmethod
    def generateSOFR():
        """Generate SOFR data and save to database"""
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

        # Save to database instead of CSV
        db = SessionLocal()
        try:
            # Clear existing data
            db.query(SOFRRate).delete()

            # Insert new data
            for _, row in rates.iterrows():
                sofr = SOFRRate(date=row["date"].date(), sofr=row["SOFR"])
                db.add(sofr)

            db.commit()
            print(f"Generated {len(rates)} SOFR records in database")
        except Exception as e:
            db.rollback()
            print(f"Error saving SOFR to database: {e}")
        finally:
            db.close()

    @staticmethod
    def generateTreasuryYields():
        """Generate treasury yields and save to database"""
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

        merged = merged.sort_values("date").drop_duplicates("date")
        merged.ffill(inplace=True)

        # Save to database instead of CSV
        db = SessionLocal()
        try:
            # Clear existing data
            db.query(TreasuryYield).delete()

            # Insert new data
            for _, row in merged.iterrows():
                yield_data = TreasuryYield(
                    date=row["date"],
                    maturity_0_5=row.get("0.5"),
                    maturity_1_0=row.get("1.0"),
                    maturity_2_0=row.get("2.0"),
                    maturity_3_0=row.get("3.0"),
                    maturity_5_0=row.get("5.0"),
                    maturity_7_0=row.get("7.0"),
                    maturity_10_0=row.get("10.0"),
                )
                db.add(yield_data)

            db.commit()
            print(f"Generated {len(merged)} treasury yield records in database")
        except Exception as e:
            db.rollback()
            print(f"Error saving treasury yields to database: {e}")
        finally:
            db.close()

    @staticmethod
    def currentRate(date: datetime):
        """Get current rate for a specific date"""
        if date < datetime.today().date():
            db = SessionLocal()
            try:
                service = DatabaseRatesService(db)
                sofr_rate = service.get_sofr_rate(date.date())
                return sofr_rate * 0.01 if sofr_rate else None
            finally:
                db.close()

        if date == datetime.today().date() or date > datetime.today().date():
            # For future dates, use treasury yields
            db = SessionLocal()
            try:
                service = DatabaseRatesService(db)
                yields = service.get_treasury_yields(datetime.today().date())
                # Return shortest maturity yield
                return yields["0.5"] if yields and yields["0.5"] else None
            finally:
                db.close()

        return None

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
        from engines.zero_rates import ZeroRatesEngine

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

        db = SessionLocal()
        try:
            service = DatabaseRatesService(db)

            # Past or on request date: use SOFR
            if target_date <= request_date:
                sofr_rate = service.get_sofr_rate(target_date)
                return sofr_rate * 0.01 if sofr_rate else None

            # Future relative to request date: use zero curve
            yields = service.get_treasury_yields(request_date)
            if not yields:
                return None

            # Build a tenor -> par yield (decimal) mapping/series
            tenor_map = {
                0.5: yields["0.5"],
                1.0: yields["1.0"],
                2.0: yields["2.0"],
                3.0: yields["3.0"],
                5.0: yields["5.0"],
                7.0: yields["7.0"],
                10.0: yields["10.0"],
            }
            yields_series = pd.Series(tenor_map).sort_index()

            # Bootstrap zero rates
            zero_list = ZeroRatesEngine.calcZeroRates(yields_series)

            # Map half-year grid to zero rates: 0.5, 1.0, ...
            tenors = [0.5 * (i + 1) for i in range(len(zero_list))]
            zero_curve = pd.Series(zero_list, index=tenors, dtype=float)

            # Year fraction from request_date to target_date (ACT/365.25)
            days = (target_date - request_date).days
            t = max(0.0, days / 365.25)
            if t == 0:
                # immediate horizon; use shortest available zero
                return float(zero_curve.iloc[0])

            # Convert zero rates to discount factors on grid
            df_data = {}
            for T, r in zero_curve.items():
                T_float = float(T)
                r_float = float(r)
                df_data[T_float] = math.exp(-r_float * T_float)
            df_grid = pd.Series(df_data, dtype=float).sort_index()

            # Interpolate ln DF (log-linear on DF), then convert back to zero rate
            grid_T = [float(x) for x in df_grid.index.to_list()]
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
        except Exception as e:
            print(f"Error getting risk-free rate: {e}")
            return None
        finally:
            db.close()

    @staticmethod
    def updateSOFR():
        """
        Incrementally update SOFR data with only the latest values.
        Updates database instead of CSV file.
        """
        try:
            db = SessionLocal()
            try:
                # Get last date from database
                last_sofr = db.query(SOFRRate).order_by(SOFRRate.date.desc()).first()
                if last_sofr:
                    start_date = last_sofr.date + timedelta(days=1)
                else:
                    start_date = datetime(2000, 1, 1).date()

                # Get new data from FRED
                new_data = fred.get_series("SOFR", observation_start=start_date)

                if new_data.empty:
                    print(f"No new SOFR data available from {start_date}")
                    return

                # Process new data
                new_df = new_data.reset_index().rename(
                    columns={"index": "date", 0: "SOFR"}
                )
                new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
                new_df = new_df.drop_duplicates(subset=["date"]).sort_values("date")
                new_df["SOFR"] = new_df["SOFR"].ffill()

                # Insert new data
                for _, row in new_df.iterrows():
                    sofr = SOFRRate(date=row["date"], sofr=row["SOFR"])
                    db.add(sofr)

                db.commit()
                print(f"Updated SOFR data: added {len(new_df)} new records")

            except Exception as e:
                db.rollback()
                print(f"Error updating SOFR data: {e}")
            finally:
                db.close()

        except Exception as e:
            print(f"Error updating SOFR data: {e}")

    @staticmethod
    def updateTreasuryYields():
        """
        Incrementally update treasury yield data with only the latest values.
        Updates database instead of CSV file.
        """
        try:
            db = SessionLocal()
            try:
                # Get last date from database
                last_yield = (
                    db.query(TreasuryYield).order_by(TreasuryYield.date.desc()).first()
                )
                if last_yield:
                    start_date = last_yield.date + timedelta(days=1)
                else:
                    start_date = datetime(2000, 1, 1).date()

                # Collect new data for each tenor
                new_data_dict = {}
                for tenor, seriesId in TENOR_TO_ID.items():
                    try:
                        new_data = fred.get_series(
                            seriesId, observation_start=start_date
                        )
                        if not new_data.empty:
                            new_df = new_data.reset_index().rename(
                                columns={"index": "date", 0: tenor}
                            )
                            new_df["parYield"] = new_df[tenor] * 0.01
                            new_df["parYield"] = new_df["parYield"].round(4)
                            new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
                            new_df = new_df.drop_duplicates(
                                subset=["date"]
                            ).sort_values("date")
                            new_df["parYield"] = new_df["parYield"].ffill()
                            new_data_dict[str(tenor)] = new_df[
                                ["date", "parYield"]
                            ].rename(columns={"parYield": str(tenor)})
                    except Exception as e:
                        print(f"Error updating tenor {tenor}: {e}")
                        continue

                if not new_data_dict:
                    print(f"No new treasury yield data available from {start_date}")
                    return

                # Merge all tenor data
                merged_new = list(new_data_dict.values())[0]
                for df in list(new_data_dict.values())[1:]:
                    merged_new = pd.merge(merged_new, df, on=["date"], how="outer")

                merged_new = merged_new.sort_values("date").drop_duplicates("date")
                merged_new = merged_new.ffill()

                # Insert new data
                for _, row in merged_new.iterrows():
                    yield_data = TreasuryYield(
                        date=row["date"],
                        maturity_0_5=row.get("0.5"),
                        maturity_1_0=row.get("1.0"),
                        maturity_2_0=row.get("2.0"),
                        maturity_3_0=row.get("3.0"),
                        maturity_5_0=row.get("5.0"),
                        maturity_7_0=row.get("7.0"),
                        maturity_10_0=row.get("10.0"),
                    )
                    db.add(yield_data)

                db.commit()
                print(f"Updated treasury yields: added {len(merged_new)} new records")

            except Exception as e:
                db.rollback()
                print(f"Error updating treasury yields: {e}")
            finally:
                db.close()

        except Exception as e:
            print(f"Error updating treasury yields: {e}")

    @staticmethod
    def updateAllRates():
        """
        Update both SOFR and treasury yield data incrementally.
        This is the main method to call for daily updates.
        """
        print("Starting incremental rate updates...")
        RatesAdapter.updateSOFR()
        RatesAdapter.updateTreasuryYields()
        print("Rate updates completed.")

    @staticmethod
    def getLastUpdateDate():
        """
        Get the last update date for both SOFR and treasury data.
        Returns a dict with 'sofr' and 'treasury' keys.
        """
        result = {}

        db = SessionLocal()
        try:
            # Check SOFR data
            last_sofr = db.query(SOFRRate).order_by(SOFRRate.date.desc()).first()
            result["sofr"] = last_sofr.date if last_sofr else None

            # Check treasury data
            last_treasury = (
                db.query(TreasuryYield).order_by(TreasuryYield.date.desc()).first()
            )
            result["treasury"] = last_treasury.date if last_treasury else None

        except Exception as e:
            print(f"Error getting last update dates: {e}")
            result = {"sofr": None, "treasury": None}
        finally:
            db.close()

        return result
