import datetime
import matplotlib.pyplot as plt
from backend.engines.IV_smile import IVEngine
from backend.adapters.options_adapter import OptionsAdapter
from backend.models.config_model import IVConfig


def fetch_options_data(cfg):
    """
    Fetch options data using the configuration.
    """
    adapter = OptionsAdapter()
    options_data = adapter.fetch_option_chain(cfg)
    return options_data


def plot_iv_smile(strikes, iv_values):
    """
    Plot the implied volatility smile.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, iv_values, marker="o", linestyle="-", color="b")
    plt.title("Implied Volatility Smile")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.show()


def main():
    ticker = "AAPL"
    today = datetime.datetime.now(datetime.timezone.utc)

    # Create IVConfig object
    cfg = IVConfig(ticker=ticker, asOf=today)

    # Fetch options data for the given configuration
    options_data = fetch_options_data(cfg)

    # Placeholder values for rate and spot
    rate = 0.05  # Example risk-free rate
    spot = 150.0  # Example spot price

    # Process the options data using the IVEngine
    iv_engine = IVEngine()
    iv_values = iv_engine.generateIVSmile(cfg, options_data, rate, spot)

    # Extract strikes and implied volatilities for plotting
    strikes = [option.strike for option in options_data]

    # Plot the IV smile
    plot_iv_smile(strikes, iv_values)


if __name__ == "__main__":
    main()
