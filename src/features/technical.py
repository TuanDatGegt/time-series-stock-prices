from __future__ import annotations

import numpy as np
import pandas as pd


def compute_return_1d(close: pd.Series) -> pd.Series:
    return close.pct_change().rename("return_1d")


def compute_return_5d(close: pd.Series) -> pd.Series:
    return close.pct_change(5).rename("return_5d")


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean().rename(f"sma_{window}")


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean().rename(f"ema_{span}")


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = pd.Series(np.where(avg_loss == 0, np.inf, avg_gain / avg_loss), index=close.index, dtype=float)
    rsi = pd.Series(np.where(avg_loss == 0, 100.0, 100 - (100 / (1 + rs))), index=close.index, dtype=float)
    return rsi.rename(f"rsi_{window}")


def compute_macd(close: pd.Series) -> pd.Series:
    ema_12 = compute_ema(close, 12)
    ema_26 = compute_ema(close, 26)
    return (ema_12 - ema_26).rename("macd")


def compute_macd_signal(macd: pd.Series) -> pd.Series:
    return macd.ewm(span=9, adjust=False, min_periods=9).mean().rename("macd_signal")


def compute_macd_hist(macd: pd.Series, signal: pd.Series) -> pd.Series:
    return (macd - signal).rename("macd_hist")


def compute_bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + (k * std)
    lower = middle - (k * std)
    return upper.rename("bb_upper"), middle.rename("bb_middle"), lower.rename("bb_lower")


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr.rename(f"atr_{window}")


def compute_volume_change(volume: pd.Series) -> pd.Series:
    return volume.pct_change().rename("volume_change")


def compute_volatility(close: pd.Series, window: int = 10) -> pd.Series:
    return close.pct_change().rolling(window=window, min_periods=window).std().rename("volatility")
