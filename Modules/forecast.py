import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from io import BytesIO
import logging
from statsmodels.tsa.stattools import adfuller, acf, pacf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_arima_params(series):
    """Determine ARIMA parameters (p, d, q) and generate ACF/PACF plots."""
    series_clean = series.dropna()
    if len(series_clean) < 2 or series_clean.std() == 0:
        logger.warning("Cannot determine ARIMA parameters: Constant or insufficient data")
        return 1, 0, 1, None  # Default values and no plot
    
    diff_count = 0
    temp_series = series_clean.copy()
    while adfuller(temp_series)[1] > 0.05 and diff_count < 2:
        temp_series = temp_series.diff().dropna()
        diff_count += 1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(temp_series, lags=20, ax=ax1)
    plot_pacf(temp_series, lags=20, ax=ax2)
    ax1.set_title("ACF")
    ax2.set_title("PACF")
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    
    # Use pacf and acf for parameter estimation
    pacf_values = pacf(temp_series, nlags=20)[1:]  # Exclude lag 0
    acf_values = acf(temp_series, nlags=20)[1:]    # Exclude lag 0
    p = next((i for i in range(len(pacf_values)) if abs(pacf_values[i]) > 0.2), 1)
    q = next((i for i in range(len(acf_values)) if abs(acf_values[i]) > 0.2), 1)
    
    # Cap p and q at 10 to match the app's max_value
    p = min(p, 10)
    q = min(q, 10)
    
    logger.info(f"ARIMA parameters determined: p={p}, d={diff_count}, q={q}")
    return p, diff_count, q, buf

def plot_prices(data, stocks, title):
    """Plot stock prices."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for stock in stocks:
        ax.plot(data.index, data[stock], label=stock)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf

def plot_pct_change(data, stocks, title):
    """Plot percentage change of stock prices."""
    pct_change = data[stocks].pct_change().dropna() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    for stock in stocks:
        ax.plot(pct_change.index, pct_change[stock], label=stock)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Percentage Change (%)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf

def plot_forecast(series, forecast, stock_name):
    """Plot historical data and forecast."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series.index[-30:], series[-30:], label="Historical", color="blue")
    forecast_index = [f"Day +{i+1}" for i in range(len(forecast))]
    ax.plot(forecast_index, forecast, label="Forecast", color="red", linestyle="--")
    ax.set_title(f"7-Day Forecast for {stock_name}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf

def forecast_stock(series, p, d, q):
    """Generate ARIMA forecast."""
    series_clean = series.dropna()
    if len(series_clean) < 2 or series_clean.std() == 0:
        raise Exception("Cannot forecast: Constant or insufficient data")
    try:
        model = ARIMA(series_clean, order=(p, d, q)).fit()
        forecast = model.forecast(steps=7)
        logger.info(f"Forecast generated for {len(forecast)} days")
        return forecast
    except Exception as e:
        logger.error(f"ARIMA forecast failed: {str(e)}")
        raise