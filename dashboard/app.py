## dashboard/app.py
"""
Module: dashboard/app.py
Description: Main Streamlit Dashboard Application.
How it works:
    Initializes Streamlit page layout, manages user sidebar selections (Symbol, Model, Date range),
    queries FastAPI endpoints via `APIClient`, computes technical features via `build_features`,
    and renders interactive Candlestick charts, RSI indicators, and prediction cards.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for feature calculations
sys.path.insert(0, str(Path(__file__).resolve().parents[8]))

from dashboard.charts import (
    create_candlestick_chart,
    create_forecast_comparison_chart,
    create_rsi_chart,
)
from dashboard.components import (
    render_header,
    render_metrics_cards,
    render_system_status,
)
from dashboard.services import APIClient
from src.features.builder import build_features

# Page Configuration
st.set_page_config(
    page_title="Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    render_header()

    # Sidebar Controls
    st.sidebar.header("🕹️ Control Panel")
    api_url = st.sidebar.text_input("API Base URL", "http://localhost:8000")
    symbol = st.sidebar.selectbox(
        "Market Symbol", ["INTC", "AAPL", "MSFT", "GOOGL"], index=0
    )
    model_name = st.sidebar.selectbox(
        "Forecasting Model", ["lstm", "gru", "xgboost"], index=0
    )

    st.sidebar.subheader("Indicators")
    show_sma = st.sidebar.checkbox("Show SMA (20/50)", value=True)
    show_ema = st.sidebar.checkbox("Show EMA (12)", value=True)

    client = APIClient(base_url=api_url)

    # 1. Health Check
    health_status = client.check_health()
    render_system_status(health_status)

    if health_status.get("status") == "offline":
        st.error(
            "⚠️ Cannot connect to FastAPI Backend. Please make sure the server is running on "
            + api_url
        )
        st.stop()

    try:
        # 2. Fetch Prediction &amp; Market Data
        pred_data = client.fetch_prediction(symbol=symbol, model=model_name)
        history_df = client.fetch_market_history(symbol=symbol)

        if history_df.empty:
            st.warning(f"No market data available for {symbol}")
            st.stop()

        # Build technical indicators for visualization
        featured_df = build_features(history_df)

        # 3. Render Top Metrics
        render_metrics_cards(pred_data)
        st.markdown("---")

        # 4. Main Charts Layout
        col_main, col_side = st.columns([0.7, 0.3])

        with col_main:
            st.subheader(f"📊 {symbol} Price Action &amp; Indicators")
            candlestick_fig = create_candlestick_chart(
                featured_df, show_sma=show_sma, show_ema=show_ema
            )
            st.plotly_chart(candlestick_fig, use_container_width=True)

            st.subheader("📉 Technical Indicators")
            rsi_fig = create_rsi_chart(featured_df)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with col_side:
            st.subheader("🎯 Model Forecast Detail")
            forecast_fig = create_forecast_comparison_chart(featured_df, pred_data)
            st.plotly_chart(forecast_fig, use_container_width=True)

            st.info(
                f"**Timestamp:** {pred_data['timestamp']}\n\n"
                f"**Price Change:** ${pred_data['price_change']:+.4f}\n\n"
                f"**Target Direction:** {pred_data['direction']}\n\n"
                f"**Model:** {pred_data['model_name'].upper()} (v{pred_data['model_version']})"
            )

    except Exception as exc:
        st.error(f"Error loading dashboard data: {exc}")


if __name__ == "__main__":
    main()
