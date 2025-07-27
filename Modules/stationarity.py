from statsmodels.tsa.stattools import adfuller
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adf_test(series, stock_name):
    """Perform ADF test for stationarity, handling constant series."""
    series_clean = series.dropna()
    if len(series_clean) < 2 or series_clean.std() == 0:
        logger.warning(f"Cannot perform ADF test on {stock_name}: Constant or insufficient data")
        return {
            "Stock": stock_name,
            "ADF Statistic": "N/A",
            "p-value": "N/A",
            "Critical Values": "N/A",
            "Stationary": "Cannot Test (Constant or Insufficient Data)"
        }
    try:
        result = adfuller(series_clean)
        logger.info(f"ADF test completed for {stock_name} with p-value {result[1]}")
        return {
            "Stock": stock_name,
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Critical Values": result[4],
            "Stationary": "Stationary" if result[1] < 0.05 else "Non-Stationary"
        }
    except Exception as e:
        logger.error(f"ADF test failed for {stock_name}: {str(e)}")
        return {
            "Stock": stock_name,
            "ADF Statistic": "N/A",
            "p-value": "N/A",
            "Critical Values": "N/A",
            "Stationary": f"Error: {str(e)}"
        }