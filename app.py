import streamlit as st
import pandas as pd
from Modules.data_loader import load_data
from Modules.stationarity import adf_test
from Modules.forecast import get_arima_params, plot_prices, plot_pct_change, plot_forecast, forecast_stock
import logging
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 5px 15px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2c3e50;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .stSubheader {
        color: #34495e;
        font-size: 1.5em;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation and settings
st.sidebar.title("Settings")
st.sidebar.markdown("Adjust your analysis preferences here.")
time_interval = st.sidebar.slider("Select time interval (days)", min_value=7, max_value=90, value=30)

# Main header
st.markdown('<div class="stHeader">Time Series Analysis & Forecasting</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#7f8c8d;">Analyze and forecast stock prices from BSE India data.</p>', unsafe_allow_html=True)

# Load data
try:
    data = load_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Stock selection in sidebar
selected_stocks = st.sidebar.multiselect("Select up to 4 stocks", options=data.columns, max_selections=4, help="Choose stocks for comparison and analysis.")

# Main content with collapsible sections
if selected_stocks:
    with st.expander("Stock Comparison", expanded=True):
        st.markdown('<div class="stSubheader">Stock Price Trends</div>', unsafe_allow_html=True)
        price_plot = plot_prices(data[selected_stocks].iloc[-time_interval:], selected_stocks, "Stock Price Comparison")
        st.image(price_plot, caption="Stock Price Trends", use_column_width=True)
        
        st.markdown('<div class="stSubheader">Percentage Change Trends</div>', unsafe_allow_html=True)
        pct_plot = plot_pct_change(data[selected_stocks].iloc[-time_interval:], selected_stocks, "Percentage Change Comparison")
        st.image(pct_plot, caption="Percentage Change Trends", use_column_width=True)
        
        st.markdown('<div class="stSubheader">Summary Statistics</div>', unsafe_allow_html=True)
        stats = data[selected_stocks].iloc[-time_interval:].describe().transpose()[["mean", "std"]]
        stats["volatility"] = stats["std"] / stats["mean"] * 100
        st.dataframe(stats.style.format("{:.2f}"))
        
        if len(selected_stocks) > 1:
            most_volatile = stats["volatility"].idxmax()
            st.info(f"**Interesting Fact**: {most_volatile} is the most volatile stock with a volatility of {stats.loc[most_volatile, 'volatility']:.2f}%.")

    with st.expander("Stationarity Tests", expanded=False):
        st.markdown('<div class="stSubheader">ADF Test Results</div>', unsafe_allow_html=True)
        adf_results = [adf_test(data[stock], stock) for stock in selected_stocks]
        adf_df = pd.DataFrame(adf_results)
        def color_stationary(val):
            if val == "Stationary":
                return 'background-color: #e6ffe6'
            elif val == "Non-Stationary":
                return 'background-color: #ffe6e6'
            return ''
        st.dataframe(adf_df.style.applymap(color_stationary, subset=['Stationary']).format({
            "ADF Statistic": "{:.2f}", "p-value": "{:.4f}"
        }))

    with st.expander("ARIMA Forecasting", expanded=False):
        stock_to_forecast = st.selectbox("Select stock for forecasting", options=selected_stocks, help="Choose a stock to predict future prices.")
        
        if stock_to_forecast:
            series = data[stock_to_forecast].dropna()
            p, d, q, acf_pacf_plot = get_arima_params(series)
            
            if acf_pacf_plot is None:
                st.warning(f"Cannot generate ARIMA parameters for {stock_to_forecast}: Constant or insufficient data")
            else:
                st.markdown('<div class="stSubheader">ACF and PACF Plots</div>', unsafe_allow_html=True)
                st.image(acf_pacf_plot, caption="ACF and PACF for ARIMA Parameter Selection", use_column_width=True)
            
            st.markdown('<div class="stSubheader">Customize ARIMA Parameters</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                p_user = st.number_input("ARIMA p (lag order)", min_value=0, max_value=10, value=p, help="Number of lag observations.")
            with col2:
                d_user = st.number_input("ARIMA d (differencing)", min_value=0, max_value=2, value=d, help="Number of differences.")
            with col3:
                q_user = st.number_input("ARIMA q (moving average)", min_value=0, max_value=10, value=q, help="Size of moving average window.")
            
            if st.button("Forecast (7 days)"):
                try:
                    forecast = forecast_stock(series, p_user, d_user, q_user)
                    forecast_index = [f"Day +{i+1}" for i in range(7)]
                    forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)
                    
                    forecast_plot = plot_forecast(series, forecast, stock_to_forecast)
                    st.image(forecast_plot, caption=f"7-Day Forecast for {stock_to_forecast}", use_column_width=True)
                    
                    st.markdown('<div class="stSubheader">Forecast Results</div>', unsafe_allow_html=True)
                    st.dataframe(forecast_df.style.format("{:.2f}"))
                except Exception as e:
                    st.error(f"Error in ARIMA model: {str(e)}")
else:
    st.warning("Please select at least one stock from the sidebar to begin analysis.")