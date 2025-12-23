## SSVI Implied Volatility Surface Construction

Arbitrage-aware implied volatility surface built using the SSVI parameterization with term-structure consistent forwards.


## Motivation

Naive interpolation of implied volatilities across strikes and maturities introduces static arbitrage and produces unstable forward dynamics. The Stochastic Volatility Inspired (SSVI) parameterization ensures a smooth, arbitrage-free volatility surface by construction, making it suitable for stable option pricing and model calibration without post-hoc corrections.


## Methodology

The implementation follows these steps:

- Bootstrapped a zero-rate curve from U.S. Treasury par yields to construct discount factors and forward prices

- Fetched options data (strikes, bid-ask, time-to-expiry) from market sources and computed log-moneyness relative to forward prices

- Computed implied volatilities from mid-prices using Black-Scholes inversion with interpolated zero rates

- Computed θ values by running an Isotonic Regression and then PCHIP interpolation to ensure monotonicity and a smooth curve

- Implemented the SSVI total variance formula with per-expiry parameters: total variance defined as w(k) = (θ/2) × [1 + ρφk + √((φk + ρ)² + 1 - ρ²)]

- Calibrated SSVI parameters phi and rho per expiry using vega-weighted least squares to match observed market implied volatilities

- Enforced no-arbitrage by constraining parameters within the SSVI feasible region (|ρ| < 1, φ > 0)

- Constructed a continuous surface across strikes and maturities 


## Interest Rate Curve

The risk free rate proxy used was rates from a boostrapped zero rate curve built from the par yield values on US treasury bonds (DGS series) ranging from 1mo to 10y. 
This is an adequate proxy for an academic project such as this, however there are a few notable downsides:

- Treasury Bonds have various effects such as liquidity premiums, safe-haven demand etc

- Zero Rates are synthetic so they may not perfectly represent the market

- Bootstrapping is a process which can introduce tiny errors which propagate

OIS Curves are a much better choice, however to get them one could either get OIS quotes directly or build the curve from SOFR futures, both of which are not freely available data and therefore beyond the scope of the project.


## Applications

This surface enables:

- Stable interpolation of implied volatilities at arbitrary strikes and expiries

- Forward-consistent option pricing without smile violations

- Input for stochastic volatility model calibration (e.g., Heston)

- Quantitative analysis of smile dynamics and term structure behavior

- Robust volatility surface visualization


## Validation

The calibrated surface is validated through:

- Monotonicity and shape stability of total variance across log-moneyness

- Smooth behavior of variance term structure across maturities

- Absence of obvious static arbitrage artifacts (mainly butterfly and calendar)

- Visual comparison of calibrated surface against raw market implied volatilities

## Evaluation

For a project of this scale, the results have been fairly successful. 
Across the MAG7 stocks:

```
Average RMSE: 0.0114
Median RMSE: 0.0099
Average Vega Weighted RMSE: 0.0088
Median Vega Weighted RMSE: 0.0079
```

This is in line with Academic papers, although the sample set my metrics were tested on consists of very well behaved highly liquid stocks, so the results may not be consistent across less trade equities.

<img width="968" height="700" alt="newplot" src="https://github.com/user-attachments/assets/0ff49916-8c81-4518-99a0-50ce61e395eb" />

Visually, the model seems to have generated a very satisfactory surface that represents the market, while being arbitrage free.

<img width="1460" height="956" alt="image" src="https://github.com/user-attachments/assets/35382c03-0f9a-4444-a746-456b3f9e4ff9" />

The residuals are also all evenly spread on both sides of 0, with no visible bias for moneyness, suggesting there are no/minimal structural problems and that the model has managed to capture the volatility smile effectively.


## Limitations & Future Work

Current limitations:

- Treasury curve used instead of OIS due to data availability

- Calibration performed slice-by-slice (per expiry) rather than jointly across all expiries

- Model parametrization does not account for volatility clustering or jump risk

Possible Future extensions:

- Implement OIS-based discounting for alignment with modern frameworks

- Joint surface calibration to enforce cross-expiry consistency

- Integration with stochastic volatility models (eg. Heston)



