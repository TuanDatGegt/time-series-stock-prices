## dashboard/components.py
"""
Module: dashboard/components.py
Description: Reusable Streamlit UI Components and Metric Cards.
How it works:
    Renders standardized metric KPI blocks (Current Price, Predicted Price, Direction),
    system health status indicators, and model information panels.
"""

from typing import Any, Dict
import streamlit as st


def render_header():
    """
    Render Dashboard Main Title Banner.
    """
    st.title("📈 Time Series Forecasting Platform")
    st.markdown("Real-time Market Analytics &amp; Deep Learning Price Predictions")


def render_system_status(status_data: Dict[str, Any]):
    """
    Render system health badge in sidebar.
    """
    status = status_data.get("status", "offline")
    if status == "ready":
        st.sidebar.success("🟢 System Status: READY")
    else:
        st.sidebar.error(f"🔴 System Status: {status.upper()}")


def render_metrics_cards(pred_data: Dict[str, Any]):
    """
    Render top-level KPI metric cards [4].
    """
    col1, col2, col3, col4 = st.columns(4)

    curr_price = pred_data["current_price"]
    pred_price = pred_data["predicted_price"]
    change_pct = pred_data["predicted_change_pct"]
    direction = pred_data["direction"]

    col1.metric("Current Price", f"${curr_price:.2f}")
    col2.metric("Predicted Price", f"${pred_price:.2f}", delta=f"{change_pct:+.2f}%")

    direction_symbol = "▲ UP" if direction == "UP" else "▼ DOWN"
    col3.metric("Expected Trend", direction_symbol)
    col4.metric(
        "Active Model",
        f"{pred_data['model_name'].upper()} v{pred_data['model_version']}",
    )
