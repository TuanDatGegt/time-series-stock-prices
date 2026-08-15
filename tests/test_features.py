import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_features

EXPECTED_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "sma_5",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "atr_14",
    "volume_change",
    "volatility",
]


def _make_sample_df(days: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=days, freq="D")
    close = np.linspace(100.0, 150.0, days)
    return pd.DataFrame(
        {
            "symbol": "INTC",
            "timestamp": timestamps,
            "open": close - 1.5,
            "high": close + 2.5,
            "low": close - 2.5,
            "close": close,
            "volume": np.linspace(1000, 4000, days),
        }
    )


def _warmup_index(column: str) -> int:
    warmup_map = {
        "return_1d": 1,
        "return_5d": 5,
        "sma_5": 5,
        "sma_20": 20,
        "sma_50": 50,
        "ema_12": 12,
        "ema_26": 26,
        "rsi_14": 14,
        "macd": 26,
        "macd_signal": 33,
        "macd_hist": 33,
        "bb_upper": 20,
        "bb_middle": 20,
        "bb_lower": 20,
        "atr_14": 14,
        "volume_change": 1,
        "volatility": 10,
    }
    return warmup_map[column]


def test_output_shape():
    df = _make_sample_df(120)
    result = build_features(df)

    assert len(result) == len(df)
    assert set(EXPECTED_FEATURE_COLUMNS).issubset(set(result.columns))

    for column in EXPECTED_FEATURE_COLUMNS:
        assert column in result.columns
        start = _warmup_index(column)
        assert result[column].iloc[start:].notna().all(), f"{column} has NaN in post-warmup data"


def test_no_future_leakage():
    df = _make_sample_df(80)
    full = build_features(df)

    target_idx = 30
    mutated = df.copy()
    mutated.loc[target_idx + 1 :, ["open", "high", "low", "close", "volume"]] = 9999.0
    mutated_features = build_features(mutated)

    for column in EXPECTED_FEATURE_COLUMNS:
        original_value = full[column].iloc[target_idx]
        mutated_value = mutated_features[column].iloc[target_idx]
        assert original_value == pytest.approx(mutated_value, nan_ok=True)


def test_incremental_consistency():
    df = _make_sample_df(90)
    full = build_features(df)

    for idx in [10, 20, 35, 50, 70]:
        prefix = df.iloc[: idx + 1].copy()
        prefix_features = build_features(prefix)

        full_row = full.iloc[idx]
        prefix_row = prefix_features.iloc[-1]

        for column in EXPECTED_FEATURE_COLUMNS:
            assert full_row[column] == pytest.approx(prefix_row[column], nan_ok=True)
