from adapters.options_adapter import OptionsAdapter
from adapters.ticker_adapter import TickerAdapter
from models.options_data import OptionsRequest, OptionType
from datetime import date
from dateutil.relativedelta import relativedelta
from engines.IV_smile import IVEngine
from engines.zero_rates import ZeroRatesEngine

import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator

from scipy.optimize import minimize


# from update_rates import updateRates
# updateRates()

adapter = OptionsAdapter()
today = date.today()
expiry_start = today + relativedelta(weeks=4)
expiry_end = today + relativedelta(weeks=56)


def ssvi_w(k, theta, phi, rho):
    w_ssvi = (
        1 / 2 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
    )
    return w_ssvi


def objective(x, theta, k, w_mkt, idx):
    rho_raw = x[0]
    rho = np.tanh(rho_raw)
    eta = x[1:]
    phi = eta[idx] / np.sqrt(np.maximum(theta, 1e-12))
    w_model = ssvi_w(k, theta, phi, rho)
    error = w_model - w_mkt
    loss = np.dot(error, error)
    return loss


def make_constraints(n_expiries):
    cons = []
    for i in range(n_expiries):
        cons.append(
            {"type": "ineq", "fun": lambda x, i=i: 2 - x[1 + i] * (1 + abs(x[0]))}
        )
    return cons


def one_pass(ticker: str, expiry_start: date, expiry_end: date):
    req = OptionsRequest(
        ticker=ticker,
        optionType=OptionType.CALL,
        expiryStart=expiry_start,
        expiryEnd=expiry_end,
        moneynessMin=0.8,
        moneynessMax=1.2,
    )
    print(f"Building SSVI for {req.ticker}")
    df = adapter.fetch_option_chain(req)
    df["rate"] = ZeroRatesEngine.interpolate_zero_rate(df, tte_col="timeToExpiry")
    base_info = TickerAdapter.fetchBasic(req.ticker)
    div = base_info.dividendYield
    spot = base_info.spot
    surface_data = IVEngine.generateIVSmile(
        df,
        div / 100,
        spot,
        OptionType.CALL,  # type: ignore
    )  # type: ignore
    surface_data.dropna(inplace=True)
    surface_data["F"] = spot * np.exp(
        (surface_data["rate"] - div / 100) * surface_data["T"]
    )
    surface_data["k"] = np.log(surface_data["K"] / surface_data["F"])
    surface_data["w"] = surface_data["iv"] ** 2 * surface_data["T"]
    thetas = surface_data.loc[
        surface_data["k"].abs().groupby(surface_data["T"]).idxmin()
    ][["T", "k", "w"]].reset_index(drop=True)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    theta_iso = iso.fit_transform(thetas["T"], thetas["w"])  # type: ignore
    theta_spline = PchipInterpolator(thetas["T"], theta_iso, extrapolate=True)
    surface_data["theta"] = theta_spline(surface_data["T"])
    vega = IVEngine._vega(
        surface_data["iv"],
        surface_data["K"],
        surface_data["T"],
        surface_data["rate"],
        div,
        surface_data["F"],
    )
    surface_data["vega"] = np.clip(vega, 1e-6, None)
    T_vals = surface_data["T"].values
    T_unique = np.sort(surface_data["T"].unique())
    T_to_index = {t: i for i, t in enumerate(T_unique)}
    idx = np.array([T_to_index[t] for t in T_vals])
    x0 = np.r_[-0.3, np.full(len(T_unique), 0.5)]
    bounds = [(-99.0, 99.0)] + [(1e-5, 2)] * len(T_unique)
    w_mkt = surface_data["w"].values
    k = surface_data["k"].values
    theta = surface_data["theta"].values
    res = minimize(
        objective,
        x0=x0,
        args=(theta, k, w_mkt, idx),
        method="SLSQP",
        bounds=bounds,
        constraints=make_constraints(len(T_unique)),
    )
    surface_data["w_ssvi"] = ssvi_w(
        k,
        theta,
        phi=res.x[1:][idx] / np.sqrt(np.maximum(theta, 1e-12)),  # type: ignore
        rho=np.tanh(res.x[0]),
    )
    surface_data["iv_ssvi"] = np.sqrt(surface_data["w_ssvi"] / surface_data["T"])
    surface_data["square_error"] = (surface_data["iv"] - surface_data["iv_ssvi"]) ** 2
    surface_data["vega_weighted_square_error"] = surface_data["square_error"] * (
        surface_data["vega"] / surface_data["vega"].sum()
    )
    rmse = np.sqrt(np.mean((surface_data["iv_ssvi"] - surface_data["iv"]) ** 2))
    surface_data["iv_error"] = (surface_data["iv_ssvi"] / surface_data["iv"] - 1) * 100
    rmse_avg_iv = rmse / np.mean(surface_data["iv"])

    return [rmse, rmse_avg_iv]


top_100_tickers = [
    "NVDA",
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "2222.SR",
    "TSM",
    "BRK.B",
    "LLY",
    "JPM",
    "WMT",
    "TCEHY",
    "V",
    "ORCL",
    "MA",
    "XOM",
    "005930.KS",
    "JNJ",
    "PLTR",
    "BAC",
    "ASML",
    "ABBV",
    "NFLX",
    "601288.SS",
    "COST",
    "MC.PA",
    "BABA",
    "1398.HK",
    "AMD",
    "HD",
    "601939.SS",
    "PG",
    "ROG.SW",
    "GE",
    "MU",
    "CSCO",
    "CVX",
    "WFC",
    "KO",
    "UNH",
    "MS",
    "AZN",
    "SAP",
    "TM",
    "IBM",
    "CAT",
    "HSBC",
    "GS",
    "000660.KS",
    "PRX.AS",
    "NVS",
    "AXP",
    "MRK",
    "601988.SS",
    "RMS.PA",
    "0857.HK",
    "NESN.SW",
    "DHR",
    "PM",
    "LIN",
    "PFE",
    "ADBE",
    "TMO",
    "NOW",
    "INTU",
    "RY",
    "PDD",
    "DIS",
    "QCOM",
    "TXN",
    "AMGN",
    "VZ",
    "HON",
    "ISRG",
    "HDFCBANK.NS",
    "NEE",
    "LOW",
    "SPGI",
    "RTX",
    "UNP",
    "SCHW",
    "SYK",
    "BKNG",
    "ELV",
    "TJX",
    "PGR",
    "UBER",
    "LRCX",
    "VRTX",
    "ETN",
    "REGN",
    "BSX",
    "PANW",
    "C",
    "MDLZ",
    "GILD",
    "CB",
]

results = {}

for ticker in top_100_tickers:
    try:
        rmse, rmse_avg_iv = one_pass(ticker, expiry_start, expiry_end)
        results[ticker] = (rmse, rmse_avg_iv)
        print(f"{ticker}: RMSE = {rmse:.4f}, RMSE/Avg IV = {rmse_avg_iv * 100:.4f}%")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

print("\nSummary of Results:")

mean_rmse = np.mean([v[0] for v in results.values()])
rmse_avg_iv_error_overall = np.mean([v[1] for v in results.values()])
median_rmse = np.median([v[0] for v in results.values()])
median_rmse_avg_iv_error = np.median([v[1] for v in results.values()])

print(f"Average RMSE across all tickers: {mean_rmse:.4f}")
print(f"Average RMSE/Avg IV across all tickers: {rmse_avg_iv_error_overall * 100:.4f}%")
print(f"Median RMSE across all tickers: {median_rmse:.4f}")
print(f"Median RMSE/Avg IV across all tickers: {median_rmse_avg_iv_error * 100:.4f}%")
