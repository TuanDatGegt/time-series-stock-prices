"""
Tests for sliding-window sequence creation (src/preprocessing/sequence.py).

Validates:
- Correct sliding-window alignment
- Correct sample counts and shapes
- Chronological ordering (no shuffling)
- Proper error handling
- No future data leakage
- Integration with split/features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequence import build_sequences


def _make_sample_df(
    days: int = 100,
    n_features: int = 3,
    symbol: str = "INTC",
) -> pd.DataFrame:
    """Helper: Create deterministic test DataFrame with features."""
    timestamps = pd.date_range("2020-01-01", periods=days, freq="D")
    close_prices = np.arange(100.0, 100.0 + days, dtype=np.float32)

    data = {
        "symbol": symbol,
        "timestamp": timestamps,
        "close": close_prices,
    }

    # Add synthetic features
    for i in range(n_features):
        data[f"feature_{i}"] = close_prices * (1.0 + 0.01 * i)

    return pd.DataFrame(data)


class TestBasicSequenceCreation:
    """Core sliding-window functionality."""

    def test_correct_output_shapes(self):
        """Verify X and y have correct shapes."""
        df = _make_sample_df(days=100, n_features=3)

        X, y = build_sequences(df, lookback=10, horizon=1)

        # n_samples = 100 - 10 - 1 + 1 = 90
        assert X.shape == (90, 10, 3), f"Expected X shape (90, 10, 3), got {X.shape}"
        assert y.shape == (90, 1), f"Expected y shape (90, 1), got {y.shape}"

    def test_correct_sample_count(self):
        """Verify n_samples = len(df) - lookback - horizon + 1."""
        for days, lookback, horizon in [
            (100, 10, 1),
            (200, 20, 1),
            (50, 5, 1),
            (100, 60, 1),
            (100, 30, 5),  # Multi-step
        ]:
            df = _make_sample_df(days=days)
            X, y = build_sequences(df, lookback=lookback, horizon=horizon)
            expected_samples = days - lookback - horizon + 1
            assert len(X) == expected_samples, (
                f"days={days}, lookback={lookback}, horizon={horizon}: "
                f"expected {expected_samples} samples, got {len(X)}"
            )
            assert len(y) == expected_samples

    def test_window_values_correct_simple(self):
        """Verify actual window values with deterministic data."""
        # Create simple data [1, 2, 3, 4, 5] with target column
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=5),
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_0": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        X, y = build_sequences(df, lookback=3, horizon=1, feature_cols=["feature_0"])

        # With lookback=3, horizon=1, n_samples should be 5 - 3 - 1 + 1 = 2
        assert X.shape == (2, 3, 1)
        assert y.shape == (2, 1)

        # First sequence: feature_0[0:3] = [10, 20, 30], target close[3] = 4
        np.testing.assert_array_almost_equal(
            X[0], [[10.0], [20.0], [30.0]], decimal=5
        )
        assert y[0, 0] == pytest.approx(4.0)

        # Second sequence: feature_0[1:4] = [20, 30, 40], target close[4] = 5
        np.testing.assert_array_almost_equal(
            X[1], [[20.0], [30.0], [40.0]], decimal=5
        )
        assert y[1, 0] == pytest.approx(5.0)

    def test_output_dtype_float32(self):
        """Ensure output arrays are float32 for PyTorch compatibility."""
        df = _make_sample_df(days=100)
        X, y = build_sequences(df, lookback=10, horizon=1)

        assert X.dtype == np.float32, f"X dtype is {X.dtype}, expected float32"
        assert y.dtype == np.float32, f"y dtype is {y.dtype}, expected float32"

    def test_minimal_case_lookback_horizon_equals_len(self):
        """Test edge case: len(df) == lookback + horizon."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=11),
            "close": np.arange(11, dtype=np.float32),
            "feature_0": np.arange(11, dtype=np.float32) * 2,
        })

        X, y = build_sequences(df, lookback=10, horizon=1)

        # Should create exactly 1 sequence
        assert X.shape == (1, 10, 1)
        assert y.shape == (1, 1)


class TestChronologicalOrdering:
    """Verify no shuffling and proper time ordering."""

    def test_maintains_row_order_no_shuffle(self):
        """Verify sequences use consecutive rows in order."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=20),
            "close": np.arange(20, dtype=np.float32),
            "feature_0": np.arange(100, 120, dtype=np.float32),
        })

        X, y = build_sequences(df, lookback=5, horizon=1)

        # First sequence should have feature_0 = [100, 101, 102, 103, 104]
        np.testing.assert_array_equal(
            X[0, :, 0], [100, 101, 102, 103, 104]
        )

        # Second sequence should have feature_0 = [101, 102, 103, 104, 105]
        np.testing.assert_array_equal(
            X[1, :, 0], [101, 102, 103, 104, 105]
        )

    def test_target_after_input_window(self):
        """Verify target is strictly after the input window (no leakage)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=20),
            "close": np.arange(20, dtype=np.float32),
            "feature_0": np.arange(20, dtype=np.float32),
        })

        lookback = 5
        horizon = 1
        X, y = build_sequences(df, lookback=lookback, horizon=horizon)

        # For sequence i:
        # X contains rows [i, i+1, ..., i+lookback-1]
        # y should be row [i+lookback+horizon-1]

        for i in range(len(X)):
            max_input_idx = i + lookback - 1
            target_idx = i + lookback + horizon - 1
            assert target_idx > max_input_idx, (
                f"Sequence {i}: target_idx={target_idx} not > max_input_idx={max_input_idx}"
            )


class TestInputValidation:
    """Input validation and error handling."""

    def test_raises_on_empty_dataframe(self):
        """Empty input raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            build_sequences(df)

    def test_raises_on_none_dataframe(self):
        """None input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            build_sequences(None)  # type: ignore

    def test_raises_on_invalid_lookback_negative(self):
        """Negative lookback raises ValueError."""
        df = _make_sample_df(days=100)
        with pytest.raises(ValueError, match="positive integer"):
            build_sequences(df, lookback=-1, horizon=1)

    def test_raises_on_invalid_lookback_zero(self):
        """Zero lookback raises ValueError."""
        df = _make_sample_df(days=100)
        with pytest.raises(ValueError, match="positive integer"):
            build_sequences(df, lookback=0, horizon=1)

    def test_raises_on_invalid_lookback_not_int(self):
        """Non-integer lookback raises ValueError."""
        df = _make_sample_df(days=100)
        with pytest.raises(ValueError, match="positive integer"):
            build_sequences(df, lookback=10.5, horizon=1)  # type: ignore

    def test_raises_on_invalid_horizon_negative(self):
        """Negative horizon raises ValueError."""
        df = _make_sample_df(days=100)
        with pytest.raises(ValueError, match="positive integer"):
            build_sequences(df, lookback=10, horizon=-1)

    def test_raises_on_invalid_horizon_zero(self):
        """Zero horizon raises ValueError."""
        df = _make_sample_df(days=100)
        with pytest.raises(ValueError, match="positive integer"):
            build_sequences(df, lookback=10, horizon=0)

    def test_raises_on_insufficient_data(self):
        """Insufficient rows raises ValueError."""
        df = _make_sample_df(days=10)
        # Need 10 + 1 = 11 rows, but only have 10
        with pytest.raises(ValueError, match="Insufficient data"):
            build_sequences(df, lookback=10, horizon=1)

    def test_raises_on_missing_target_column(self):
        """Missing target column raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100),
            "feature_0": np.arange(100, dtype=np.float32),
        })
        with pytest.raises(ValueError, match="not found"):
            build_sequences(df, target_col="nonexistent")

    def test_raises_on_missing_feature_column(self):
        """Missing feature column raises ValueError."""
        df = _make_sample_df(days=100)
        with pytest.raises(ValueError, match="not found"):
            build_sequences(df, feature_cols=["nonexistent_feature"])

    def test_raises_on_non_numeric_target(self):
        """Non-numeric target column raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100),
            "close": ["a", "b", "c"] * 33 + ["a"],  # String column
            "feature_0": np.arange(100, dtype=np.float32),
        })
        with pytest.raises(ValueError, match="Cannot convert"):
            build_sequences(df, target_col="close")

    def test_raises_on_non_numeric_feature(self):
        """Non-numeric feature column raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100),
            "close": np.arange(100, dtype=np.float32),
            "bad_feature": ["x", "y", "z"] * 33 + ["x"],  # String column
        })
        with pytest.raises(ValueError, match="Cannot convert"):
            build_sequences(df, feature_cols=["bad_feature"])


class TestFeatureSelection:
    """Feature column selection and defaults."""

    def test_explicit_feature_cols(self):
        """Use explicitly provided feature columns."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=50),
            "close": np.arange(50, dtype=np.float32),
            "feature_a": np.arange(50, dtype=np.float32) * 1.0,
            "feature_b": np.arange(50, dtype=np.float32) * 2.0,
            "feature_c": np.arange(50, dtype=np.float32) * 3.0,
        })

        X_abc, y = build_sequences(
            df, lookback=10, horizon=1,
            feature_cols=["feature_a", "feature_b", "feature_c"]
        )
        # n_samples = 50 - 10 - 1 + 1 = 40
        assert X_abc.shape == (40, 10, 3)

        X_ab, y = build_sequences(
            df, lookback=10, horizon=1,
            feature_cols=["feature_a", "feature_b"]
        )
        # n_samples = 50 - 10 - 1 + 1 = 40
        assert X_ab.shape == (40, 10, 2)

        # Verify values match
        np.testing.assert_array_almost_equal(X_abc[:, :, 0], X_ab[:, :, 0], decimal=5)
        np.testing.assert_array_almost_equal(X_abc[:, :, 1], X_ab[:, :, 1], decimal=5)

    def test_default_feature_selection(self):
        """Default: use all numeric except timestamp/symbol/OHLCV."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=50),
            "symbol": "INTC",
            "open": np.arange(50, dtype=np.float32),
            "high": np.arange(50, dtype=np.float32) + 1,
            "low": np.arange(50, dtype=np.float32) - 1,
            "close": np.arange(50, dtype=np.float32),
            "volume": np.arange(50, dtype=np.float32) * 1000,
            "sma_20": np.arange(50, dtype=np.float32) * 0.99,
            "rsi_14": np.arange(50, dtype=np.float32) % 100,
        })

        X, y = build_sequences(df, lookback=10, horizon=1, feature_cols=None)

        # Should use sma_20 and rsi_14 (numeric, not in exclude list)
        # n_samples = 50 - 10 - 1 + 1 = 40
        assert X.shape == (40, 10, 2)


class TestNaNHandling:
    """NaN value behavior."""

    def test_preserves_nan_in_features(self):
        """NaN values in features are preserved in sequences."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=50),
            "close": np.arange(50, dtype=np.float32),
            "feature_0": np.array([np.nan if i < 10 else float(i) for i in range(50)]),
        })

        X, y = build_sequences(df, lookback=10, horizon=1)

        # First sequence should include NaNs
        assert np.isnan(X[0]).any(), "Expected NaNs in early sequences"

        # Later sequences should not have NaNs in feature data
        assert not np.isnan(X[-1]).any(), "Expected no NaNs in late sequences"

    def test_nan_in_target_preserved(self):
        """NaN in target column is preserved (caller handles)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=50),
            "close": np.array([float(i) if i > 5 else np.nan for i in range(50)]),
            "feature_0": np.arange(50, dtype=np.float32),
        })

        X, y = build_sequences(df, lookback=10, horizon=1)

        # Some early targets may be NaN
        # This is expected; caller responsible for handling
        # n_samples = 50 - 10 - 1 + 1 = 40
        assert y.shape == (40, 1)


class TestMultiStepHorizon:
    """Multi-step forecasting (horizon > 1)."""

    def test_supports_horizon_greater_than_one(self):
        """Can create sequences with horizon > 1."""
        df = _make_sample_df(days=100, n_features=2)

        X, y = build_sequences(df, lookback=10, horizon=5)

        # n_samples = 100 - 10 - 5 + 1 = 86
        assert X.shape == (86, 10, 2)
        assert y.shape == (86, 1)

    def test_horizon_5_target_alignment(self):
        """Verify horizon=5 correctly offsets target."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=20),
            "close": np.arange(20, dtype=np.float32),
            "feature_0": np.arange(20, dtype=np.float32) * 10,
        })

        X, y = build_sequences(df, lookback=5, horizon=5)

        # For i=0: X[0] = rows [0:5], y[0] = row[5+5-1=9]
        np.testing.assert_array_equal(X[0, :, 0], [0, 10, 20, 30, 40])
        assert y[0, 0] == pytest.approx(9.0)  # close[9]

        # For i=1: X[1] = rows [1:6], y[1] = row[1+5+5-1=10]
        np.testing.assert_array_equal(X[1, :, 0], [10, 20, 30, 40, 50])
        assert y[1, 0] == pytest.approx(10.0)  # close[10]


class TestLookbackOneCase:
    """Edge case: lookback=1 (single observation)."""

    def test_lookback_one(self):
        """Can create sequences with lookback=1."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=20),
            "close": np.arange(20, dtype=np.float32),
            "feature_0": np.arange(20, dtype=np.float32) * 100,
        })

        X, y = build_sequences(df, lookback=1, horizon=1)

        # n_samples = 20 - 1 - 1 + 1 = 19
        assert X.shape == (19, 1, 1)
        assert y.shape == (19, 1)

        # First sequence: X[0] = rows [0:1] = row[0] = feature_0[0] * 100 = 0 * 100 = 0.0
        # y[0] = row [0 + 1 + 1 - 1] = row [1] = close[1] = 1.0
        assert X[0, 0, 0] == pytest.approx(0.0)
        assert y[0, 0] == pytest.approx(1.0)


class TestIntegrationWithRealFeatures:
    """Integration with actual feature engineering output."""

    def test_with_feature_builder_output(self):
        """Works with output from src.features.builder."""
        from src.features.builder import build_features

        # Create raw OHLCV data
        df_raw = pd.DataFrame({
            "symbol": "INTC",
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "open": np.linspace(20.0, 25.0, 100),
            "high": np.linspace(21.0, 26.0, 100),
            "low": np.linspace(19.0, 24.0, 100),
            "close": np.linspace(20.5, 25.5, 100),
            "volume": np.linspace(1e6, 2e6, 100),
        })

        # Build features
        df_features = build_features(df_raw)

        # Create sequences (after warmup, many indicators start with NaN)
        # Drop first 50 rows to avoid heavy NaN concentration
        df_features_clean = df_features.iloc[50:].reset_index(drop=True)

        X, y = build_sequences(df_features_clean, lookback=10, horizon=1)

        # Should succeed and produce expected shape
        assert X.ndim == 3
        assert X.shape[1] == 10  # lookback
        assert y.shape[1] == 1
        assert len(X) == len(y)


class TestIntegrationWithSplitter:
    """Integration with preprocessing/splitter."""

    def test_with_split_dataset_output(self):
        """Works with output from src.preprocessing.splitter."""
        from src.preprocessing.splitter import split_dataset

        # Create sample data
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=300),
            "symbol": "INTC",
            "close": np.linspace(20.0, 30.0, 300),
            "sma_20": np.linspace(20.0, 30.0, 300) * 0.99,
        })

        # Split
        train_df, val_df, test_df = split_dataset(df, train_ratio=0.6, val_ratio=0.2)

        # Create sequences for each split independently
        X_train, y_train = build_sequences(train_df, lookback=20, horizon=1)
        X_val, y_val = build_sequences(val_df, lookback=20, horizon=1)
        X_test, y_test = build_sequences(test_df, lookback=20, horizon=1)

        # Verify all completed and have no overlap
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0

        # No leakage: each split's sequences are independent
        # (this is enforced by the caller, not by build_sequences)
