# src/features/builder.py

from __future__ import annotations

import pandas as pd

from src.features.technical import (
    compute_atr,
    compute_bollinger,
    compute_ema,
    compute_macd,
    compute_macd_hist,
    compute_macd_signal,
    compute_return_1d,
    compute_return_5d,
    compute_rsi,
    compute_sma,
    compute_volatility,
    compute_volume_change,
)


def _build_features_for_group(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("timestamp").copy()
    close = ordered["close"].astype(float)
    high = ordered["high"].astype(float)
    low = ordered["low"].astype(float)
    volume = ordered["volume"].astype(float)

    ordered["return_1d"] = compute_return_1d(close)
    ordered["return_5d"] = compute_return_5d(close)
    ordered["sma_5"] = compute_sma(close, 5)
    ordered["sma_20"] = compute_sma(close, 20)
    ordered["sma_50"] = compute_sma(close, 50)
    ordered["ema_12"] = compute_ema(close, 12)
    ordered["ema_26"] = compute_ema(close, 26)
    ordered["rsi_14"] = compute_rsi(close, 14)

    macd = compute_macd(close)
    ordered["macd"] = macd
    ordered["macd_signal"] = compute_macd_signal(macd)
    ordered["macd_hist"] = compute_macd_hist(macd, ordered["macd_signal"])

    bb_upper, bb_middle, bb_lower = compute_bollinger(close, 20, 2.0)
    ordered["bb_upper"] = bb_upper
    ordered["bb_middle"] = bb_middle
    ordered["bb_lower"] = bb_lower

    ordered["atr_14"] = compute_atr(high, low, close, 14)
    ordered["volume_change"] = compute_volume_change(volume)
    ordered["volatility"] = compute_volatility(close, 10)

    return ordered


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    working = df.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"])
    working = working.sort_values("timestamp").reset_index(drop=True)

    if "symbol" in working.columns:
        result = (
            working.groupby("symbol", group_keys=False)
            .apply(_build_features_for_group)
            .reset_index(drop=True)
        )
        return result.sort_values("timestamp").reset_index(drop=True)

    return _build_features_for_group(working).reset_index(drop=True)
