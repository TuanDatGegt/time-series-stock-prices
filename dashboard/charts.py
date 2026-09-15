## dashboard/charts.py
"""
Module: dashboard/charts.py
Description: Interactive Plotly Chart Builders for Financial Time Series.
How it works:
    Constructs multi-panel interactive Plotly charts including Candlestick price action,
    Volume bars, Moving Average overlays (SMA/EMA), and Technical Indicators (RSI, MACD).
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df: pd.DataFrame,
    show_sma: bool = True,
    show_ema: bool = True,
) -> go.Figure:
    """
    Build 2-panel Plotly figure: Price Candlesticks with MAs (top) and Volume bars (bottom).
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price Action &amp; Indicators", "Volume"),
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # 2. Moving Average Overlays (if columns exist)
    if show_sma:
        if "sma_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["sma_20"],
                    mode="lines",
                    name="SMA 20",
                    line=dict(color="orange", width=1.5),
                ),
                row=1,
                col=1,
            )
        if "sma_50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["sma_50"],
                    mode="lines",
                    name="SMA 50",
                    line=dict(color="blue", width=1.5),
                ),
                row=1,
                col=1,
            )

    if show_ema and "ema_12" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ema_12"],
                mode="lines",
                name="EMA 12",
                line=dict(color="purple", width=1.5),
            ),
            row=1,
            col=1,
        )

    # 3. Volume Bar Trace
    colors = [
        "green" if close >= open_val else "red"
        for close, open_val in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """
    Build standalone RSI (Relative Strength Index) technical indicator chart.
    """
    fig = go.Figure()
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["rsi_14"],
                mode="lines",
                name="RSI 14",
                line=dict(color="cyan", width=1.5),
            )
        )
        fig.add_hline(
            y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)"
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)"
        )

    fig.update_layout(
        title="Relative Strength Index (RSI 14)",
        template="plotly_dark",
        height=250,
        yaxis=dict(range=[5]),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_forecast_comparison_chart(df: pd.DataFrame, prediction: dict) -> go.Figure:
    """
    Build actual price series with target forecast point overlay.
    """
    fig = go.Figure()

    # Plot recent actual closing prices
    recent_df = df.tail(30).copy()
    fig.add_trace(
        go.Scatter(
            x=recent_df["timestamp"],
            y=recent_df["close"],
            mode="lines+markers",
            name="Actual Close",
            line=dict(color="white", width=2),
        )
    )

    # Add forecast point
    last_time = recent_df["timestamp"].iloc[-1]
    next_time = last_time + pd.Timedelta(days=1)
    pred_price = prediction["predicted_price"]

    fig.add_trace(
        go.Scatter(
            x=[last_time, next_time],
            y=[recent_df["close"].iloc[-1], pred_price],
            mode="lines+markers",
            name="Forecast Target",
            line=dict(
                color="green" if prediction["direction"] == "UP" else "red",
                width=2,
                dash="dash",
            ),
            marker=dict(size=10, symbol="star"),
        )
    )

    fig.update_layout(
        title=f"Next-Day Price Projection ({prediction['model_name'].upper()})",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
