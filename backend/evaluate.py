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
expiry_start = today + relativedelta(weeks=5)
expiry_end = today + relativedelta(weeks=56)


def ssvi_w(k, theta, phi, rho):
    w_ssvi = (
        1 / 2 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
    )
    return w_ssvi


def objective(x, theta, k, w_mkt, idx, weights):
    rho_raw = x[0]
    rho = np.tanh(rho_raw)
    eta = x[1:]
    phi = eta[idx] / np.sqrt(np.maximum(theta, 1e-12))
    w_model = ssvi_w(k, theta, phi, rho)
    error = w_model - w_mkt
    loss = np.dot(weights * error, error) * 100000
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
    weights = surface_data.groupby("T")["vega"].transform(lambda x: x / x.sum())
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
        args=(theta, k, w_mkt, idx, weights),
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

    rmse = np.sqrt(np.mean((surface_data["iv_ssvi"] - surface_data["iv"]) ** 2))

    vega_weighted_rmse = np.sqrt(
        np.sum(
            ((surface_data["iv_ssvi"] - surface_data["iv"]) ** 2)
            * (surface_data["vega"] / np.sum(surface_data["vega"]))
        )
    )

    return [rmse, vega_weighted_rmse]


mag7_tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "TSLA",
    "META",
]

results = {}

for ticker in mag7_tickers:
    try:
        rmse, vega_weighted_rmse = one_pass(ticker, expiry_start, expiry_end)
        results[ticker] = (rmse, vega_weighted_rmse)
        print(
            f"{ticker}: RMSE = {rmse:.4f}, Vega Weighted RMSE = {vega_weighted_rmse:.4f}"
        )
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

print("\nSummary of Results:")

mean_rmse = np.mean([v[0] for v in results.values()])
vega_weighted_rmse_overall = np.mean([v[1] for v in results.values()])
median_rmse = np.median([v[0] for v in results.values()])
median_vega_weighted_rmse = np.median([v[1] for v in results.values()])

print(f"Average RMSE across all tickers: {mean_rmse:.4f}")
print(
    f"Average Vega Weighted RMSE across all tickers: {vega_weighted_rmse_overall:.4f}"
)
print(f"Median RMSE across all tickers: {median_rmse:.4f}")
print(f"Median Vega Weighted RMSE across all tickers: {median_vega_weighted_rmse:.4f}")
