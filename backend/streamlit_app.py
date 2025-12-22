import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

from adapters.options_adapter import OptionsAdapter
from adapters.ticker_adapter import TickerAdapter
from models.options_data import OptionsRequest, OptionType
from engines.IV_smile import IVEngine
from engines.zero_rates import ZeroRatesEngine
from update_rates import updateRates

# Set page config
st.set_page_config(page_title="Options IV Analysis", layout="wide")

# Title
st.title("Options Implied Volatility Analysis")
st.markdown("This app analyzes options implied volatility and fits the SSVI model.")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Ticker input
ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA")

# Duration selection
duration_months = st.sidebar.slider(
    "Duration (months)", min_value=1, max_value=60, value=12
)

# Moneyness range
col1, col2 = st.sidebar.columns(2)
with col1:
    moneyness_min = st.number_input("Moneyness Min", value=0.8, step=0.1)
with col2:
    moneyness_max = st.number_input("Moneyness Max", value=1.2, step=0.1)

# Button to run analysis
run_analysis = st.sidebar.button("Run Analysis", type="primary")

if run_analysis:
    # Add a status container
    status_container = st.container()

    with status_container:
        st.info("🔄 Running analysis... This may take a moment.")

        try:
            # Update rates
            with st.spinner("Updating rates..."):
                updateRates()

            # Setup adapter and request
            adapter = OptionsAdapter()
            today = date.today()
            expiry_start = today + relativedelta(weeks=4)
            expiry_end = today + relativedelta(months=duration_months)

            req = OptionsRequest(
                ticker=ticker,
                optionType=OptionType.CALL,
                expiryStart=expiry_start,
                expiryEnd=expiry_end,
                moneynessMin=moneyness_min,
                moneynessMax=moneyness_max,
            )

            # Fetch options data
            with st.spinner("Fetching options data..."):
                df = adapter.fetch_option_chain(req)

            st.success(f"✓ Fetched {len(df)} contracts")

            # Calculate IV
            with st.spinner("Calculating implied volatility..."):
                df["rate"] = ZeroRatesEngine.interpolate_zero_rate(
                    df, tte_col="timeToExpiry"
                )

                base_info = TickerAdapter.fetchBasic(req.ticker)
                div = base_info.dividendYield
                spot = base_info.spot

                surface_data = IVEngine.generateIVSmile(
                    df, df["rate"], div / 100, spot, OptionType.CALL
                )
                surface_data.dropna(inplace=True)

            st.success(f"✓ Calculated IV for {len(surface_data)} data points")

            # Display basic info
            col1, col2, col3 = st.columns(3)
            col1.metric("Spot Price", f"${spot:.2f}")
            col2.metric("Dividend Yield", f"{div:.2f}%")
            col3.metric("Data Points", len(surface_data))

            # Prepare SSVI data
            surface_data["F"] = spot * np.exp(
                (surface_data["rate"] - div / 100) * surface_data["T"]
            )
            surface_data["k"] = np.log(surface_data["K"] / surface_data["F"])
            surface_data["w"] = surface_data["iv"] ** 2 * surface_data["T"]

            # Calculate theta
            theta_low = (
                surface_data[surface_data["k"] < 0]
                .groupby("T")[["expiry", "k", "w", "K", "F", "iv"]]
                .max()
            )
            theta_high = (
                surface_data[surface_data["k"] > 0]
                .groupby("T")[["expiry", "k", "w", "K", "F", "iv"]]
                .min()
            )
            interpolated_values = theta_high["w"] - (
                theta_high["w"] - theta_low["w"]
            ) * (theta_high["k"] / (theta_high["k"] - theta_low["k"]))
            surface_data["theta"] = surface_data["T"].map(interpolated_values)

            # Calculate vega
            vega = IVEngine._vega(
                surface_data["iv"],
                surface_data["K"],
                surface_data["T"],
                surface_data["rate"],
                div,
                surface_data["F"],
            )
            surface_data["vega"] = np.clip(vega, 1e-6, None)

            # SSVI calibration
            with st.spinner("Calibrating SSVI model..."):
                T_vals = surface_data["T"].values
                T_unique = np.sort(surface_data["T"].unique())
                T_to_index = {t: i for i, t in enumerate(T_unique)}
                idx = np.array([T_to_index[t] for t in T_vals])
                x0 = np.r_[-0.3, np.full(len(T_unique), 0.5)]

                def ssvi_w(k, theta, phi, rho):
                    w_ssvi = (
                        1
                        / 2
                        * theta
                        * (
                            1
                            + rho * phi * k
                            + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2)
                        )
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
                            {
                                "type": "ineq",
                                "fun": lambda x, i=i: 2 - x[1 + i] * (1 + abs(x[0])),
                            }
                        )
                    return cons

                bounds = [(-5.0, 5.0)] + [(1e-5, 2)] * len(T_unique)
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

                rho_opt = np.tanh(res.x[0])
                eta_opt = res.x[1:]
                ssvi_loss = objective(res.x, theta, k, w_mkt, idx)

                surface_data["w_ssvi"] = ssvi_w(
                    k,
                    theta,
                    phi=eta_opt[idx] / np.sqrt(np.maximum(theta, 1e-12)),
                    rho=rho_opt,
                )

            st.success("✓ SSVI calibration complete")

            # Display SSVI parameters
            col1, col2 = st.columns(2)
            col1.metric("Optimized Rho", f"{rho_opt:.6f}")
            col2.metric("SSVI Loss", f"{ssvi_loss:.6f}")

            # Calculate errors
            surface_data["residuals"] = surface_data["w_ssvi"] - surface_data["w"]
            surface_data["relative_residuals"] = (
                surface_data["residuals"] / surface_data["w"]
            ) * 100

            surface_data["iv_ssvi"] = np.sqrt(
                surface_data["w_ssvi"] / surface_data["T"]
            )
            surface_data["iv_error"] = (
                surface_data["iv_ssvi"] / surface_data["iv"] - 1
            ) * 100

            # Display data tables
            st.header("Data Summary")

            tab1, tab2, tab3 = st.tabs(["Options Data", "Surface Data", "SSVI Results"])

            with tab1:
                st.dataframe(
                    df[
                        ["optionType", "strike", "timeToExpiry", "midPrice", "expiry"]
                    ].head(20),
                    use_container_width=True,
                    height=400,
                )

            with tab2:
                st.dataframe(
                    surface_data[["expiry", "K", "k", "w", "theta"]].head(20),
                    use_container_width=True,
                    height=400,
                )

            with tab3:
                st.dataframe(
                    surface_data[["expiry", "K", "iv", "iv_ssvi", "iv_error"]].head(20),
                    use_container_width=True,
                    height=400,
                )

            # Visualizations
            st.header("Visualizations")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("SSVI Relative Residuals")
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    surface_data["k"],
                    surface_data["relative_residuals"],
                    c=surface_data["T"],
                    cmap="viridis",
                    alpha=0.6,
                    s=50,
                )
                plt.colorbar(scatter, ax=ax, label="Time to Expiry (T)")
                ax.set_xlabel("Log-Moneyness (k)")
                ax.set_ylabel("Relative Residuals (%)")
                ax.set_title("SSVI Model Relative Residuals vs Log-Moneyness")
                ax.axhline(0, color="red", linestyle="--", linewidth=2)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("IV Error")
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    surface_data["k"],
                    surface_data["iv_error"],
                    c=surface_data["T"],
                    cmap="viridis",
                    alpha=0.6,
                    s=50,
                )
                plt.colorbar(scatter, ax=ax, label="Time to Expiry (T)")
                ax.set_xlabel("Log-Moneyness (k)")
                ax.set_ylabel("Implied Volatility Error (%)")
                ax.set_title("IV Error vs Log-Moneyness")
                ax.axhline(0, color="red", linestyle="--", linewidth=2)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            # Additional metrics
            st.header("Performance Metrics")

            mean_relative_error = np.mean(np.abs(surface_data["relative_residuals"]))
            max_relative_error = np.max(np.abs(surface_data["relative_residuals"]))
            mean_iv_error = np.mean(np.abs(surface_data["iv_error"]))
            max_iv_error = np.max(np.abs(surface_data["iv_error"]))

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Relative Error", f"{mean_relative_error:.4f}%")
            col2.metric("Max Relative Error", f"{max_relative_error:.4f}%")
            col3.metric("Mean IV Error", f"{mean_iv_error:.4f}%")
            col4.metric("Max IV Error", f"{max_iv_error:.4f}%")

            st.success("✓ Analysis complete!")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback

            st.error(traceback.format_exc())
else:
    st.info(
        "👈 Configure the parameters in the sidebar and click 'Run Analysis' to begin."
    )
