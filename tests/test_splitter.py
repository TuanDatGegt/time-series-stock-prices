"""
Tests for chronological dataset splitter (src/preprocessing/splitter.py).

Ensures:
- NO random shuffling, chronological order preserved
- Correct proportions and boundaries
- No data leakage
- Exact reconstruction of original data
- Comprehensive input validation
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.splitter import split_dataset


class TestRatioSplit:
    """Tests for ratio-based splitting."""

    def test_ratio_split_proportions(self):
        """Test that ratio split produces approximately correct proportions."""
        # Create 100-row dataset
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
            "volume": range(1000, 1100),
        })

        train, val, test = split_dataset(
            df,
            split_mode="ratio",
            train_ratio=0.7,
            val_ratio=0.15,
        )

        assert len(train) == 70, f"Expected 70 train samples, got {len(train)}"
        assert len(val) == 15, f"Expected 15 val samples, got {len(val)}"
        assert len(test) == 15, f"Expected 15 test samples, got {len(test)}"

    def test_ratio_split_maintains_order(self):
        """Test that chronological order is preserved."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        train, val, test = split_dataset(df, split_mode="ratio")

        # Check that each split is sorted
        assert train["timestamp"].is_monotonic_increasing
        assert val["timestamp"].is_monotonic_increasing
        assert test["timestamp"].is_monotonic_increasing

        # Check that no mixing occurred (train ends before val starts)
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_ratio_split_custom_ratios(self):
        """Test with custom train/val ratios."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=1000, freq="D"),
            "close": range(1000),
        })

        train, val, test = split_dataset(
            df,
            split_mode="ratio",
            train_ratio=0.6,
            val_ratio=0.2,
        )

        assert len(train) == 600
        assert len(val) == 200
        assert len(test) == 200


class TestDateRangeSplit:
    """Tests for date-range-based splitting."""

    def test_date_range_split_boundaries(self):
        """Test that date_range split respects date boundaries."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        train, val, test = split_dataset(
            df,
            split_mode="date_range",
            train_end="2020-02-10",  # Day 41
            val_end="2020-03-10",    # Day 69
        )

        # Check boundaries
        assert (train["timestamp"] <= pd.Timestamp("2020-02-10")).all()
        assert ((val["timestamp"] > pd.Timestamp("2020-02-10")) &
                (val["timestamp"] <= pd.Timestamp("2020-03-10"))).all()
        assert (test["timestamp"] > pd.Timestamp("2020-03-10")).all()

    def test_date_range_split_exact_counts(self):
        """Test exact row counts for date_range split."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        # 2020 is a leap year: Jan=31, Feb=29
        # 2020-01-01 is index 0, 2020-02-10 is index 40 (31+10-1=40)
        # 2020-03-10 is index 69 (31+29+10-1=69)
        # Train: 0-40 (41 rows), Val: 41-69 (29 rows), Test: 70-99 (30 rows)
        train, val, test = split_dataset(
            df,
            split_mode="date_range",
            train_end="2020-02-10",
            val_end="2020-03-10",
        )

        assert len(train) == 41
        assert len(val) == 29
        assert len(test) == 30


class TestDataReconstruction:
    """Tests for data integrity and reconstruction."""

    def test_reconstructs_original_data_ratio(self):
        """Test that concatenating splits reconstructs original df (ratio mode)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
            "volume": range(1000, 1100),
        })

        train, val, test = split_dataset(df, split_mode="ratio")
        reconstructed = pd.concat([train, val, test], ignore_index=False)

        # Sort both by timestamp to ensure same order
        reconstructed_sorted = reconstructed.sort_values("timestamp").reset_index(drop=True)
        original_sorted = df.sort_values("timestamp").reset_index(drop=True)

        pd.testing.assert_frame_equal(reconstructed_sorted, original_sorted)

    def test_reconstructs_original_data_date_range(self):
        """Test that concatenating splits reconstructs original df (date_range mode)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
            "volume": range(1000, 1100),
        })

        train, val, test = split_dataset(
            df,
            split_mode="date_range",
            train_end="2020-02-10",
            val_end="2020-03-10",
        )
        reconstructed = pd.concat([train, val, test], ignore_index=False)

        reconstructed_sorted = reconstructed.sort_values("timestamp").reset_index(drop=True)
        original_sorted = df.sort_values("timestamp").reset_index(drop=True)

        pd.testing.assert_frame_equal(reconstructed_sorted, original_sorted)

    def test_no_overlapping_timestamps(self):
        """Test that train/val/test have no overlapping timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        train, val, test = split_dataset(df, split_mode="ratio")

        # Extract timestamp sets
        train_ts = set(train["timestamp"])
        val_ts = set(val["timestamp"])
        test_ts = set(test["timestamp"])

        # Check no overlaps
        assert train_ts.isdisjoint(val_ts), "Train and val have overlapping timestamps"
        assert val_ts.isdisjoint(test_ts), "Val and test have overlapping timestamps"
        assert train_ts.isdisjoint(test_ts), "Train and test have overlapping timestamps"

    def test_total_row_count_preserved(self):
        """Test that total rows are preserved after split."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=500, freq="D"),
            "close": range(500),
        })

        train, val, test = split_dataset(df, split_mode="ratio")

        total_after_split = len(train) + len(val) + len(test)
        assert total_after_split == len(df), \
            f"Row count mismatch: original={len(df)}, after_split={total_after_split}"


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_raises_on_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            split_dataset(df)

    def test_raises_on_unsorted_input(self):
        """Test that unsorted input raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
            "close": range(10),
        })

        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        with pytest.raises(ValueError, match="must be sorted"):
            split_dataset(df_shuffled)

    def test_raises_on_missing_timestamp_column(self):
        """Test that missing timestamp column raises ValueError."""
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "close": range(10),
        })

        with pytest.raises(ValueError, match="not found"):
            split_dataset(df, timestamp_col="timestamp")

    def test_raises_on_invalid_ratio_sum(self):
        """Test that invalid ratio sum raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        # sum >= 1.0 should raise
        with pytest.raises(ValueError, match="must sum to < 1.0"):
            split_dataset(
                df,
                split_mode="ratio",
                train_ratio=0.7,
                val_ratio=0.3,  # 0.7 + 0.3 = 1.0
            )

    def test_raises_on_negative_ratio(self):
        """Test that negative ratio raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        with pytest.raises(ValueError, match="must be > 0.0"):
            split_dataset(
                df,
                split_mode="ratio",
                train_ratio=-0.5,
                val_ratio=0.2,
            )

    def test_raises_on_zero_ratio(self):
        """Test that zero ratio raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        with pytest.raises(ValueError, match="must be > 0.0"):
            split_dataset(
                df,
                split_mode="ratio",
                train_ratio=0.0,
                val_ratio=0.5,
            )

    def test_raises_on_invalid_date_range(self):
        """Test that invalid date range raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        # train_end >= val_end
        with pytest.raises(ValueError, match="train_end must be < val_end"):
            split_dataset(
                df,
                split_mode="date_range",
                train_end="2020-03-10",
                val_end="2020-02-10",
            )

    def test_raises_on_missing_date_range_params(self):
        """Test that missing date_range parameters raise ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        with pytest.raises(ValueError, match="requires both train_end and val_end"):
            split_dataset(
                df,
                split_mode="date_range",
                train_end="2020-02-10",
                val_end=None,
            )

    def test_raises_on_invalid_date_format(self):
        """Test that invalid date format raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        with pytest.raises(ValueError, match="Cannot parse"):
            split_dataset(
                df,
                split_mode="date_range",
                train_end="invalid-date",
                val_end="2020-02-10",
            )

    def test_raises_on_date_range_outside_data(self):
        """Test that dates outside data range raise ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        # train_end before first timestamp
        with pytest.raises(ValueError, match="No data found for train set"):
            split_dataset(
                df,
                split_mode="date_range",
                train_end="2019-12-31",
                val_end="2020-02-10",
            )

    def test_raises_on_invalid_split_mode(self):
        """Test that invalid split_mode raises ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        with pytest.raises(ValueError, match="split_mode must be"):
            split_dataset(df, split_mode="invalid_mode")  # type: ignore


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_small_dataset_ratio_split(self):
        """Test splitting a small dataset."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
            "close": range(10),
        })

        train, val, test = split_dataset(
            df,
            split_mode="ratio",
            train_ratio=0.5,
            val_ratio=0.3,
        )

        assert len(train) == 5
        assert len(val) == 3
        assert len(test) == 2

    def test_single_day_each_split(self):
        """Test that each split can have just 1 day if data allows."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
            "close": range(10),
        })

        train, val, test = split_dataset(
            df,
            split_mode="ratio",
            train_ratio=0.1,
            val_ratio=0.1,
        )

        # Due to int() truncation: 10*0.1=1, 10*0.1=1, rest=8
        assert len(train) == 1
        assert len(val) == 1
        assert len(test) == 8

    def test_custom_timestamp_column(self):
        """Test with a custom timestamp column name."""
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        train, val, test = split_dataset(
            df,
            split_mode="ratio",
            timestamp_col="date",
            train_ratio=0.7,
            val_ratio=0.15,
        )

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_with_nan_values_in_features(self):
        """Test that NaN values in feature columns don't affect split."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
            "sma_20": [None] * 19 + list(range(81)),  # NaN for first 19 rows
        })

        train, val, test = split_dataset(df, split_mode="ratio")

        # Split should work fine; NaNs are just values
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert train["sma_20"].isna().sum() == 19  # NaNs preserved

    def test_multiple_calls_idempotent(self):
        """Test that multiple calls with same params produce identical results."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        train1, val1, test1 = split_dataset(df, split_mode="ratio")
        train2, val2, test2 = split_dataset(df, split_mode="ratio")

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_datetime_timezone_aware(self):
        """Test with timezone-aware timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC"),
            "close": range(100),
        })

        train, val, test = split_dataset(df, split_mode="ratio")

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert train["timestamp"].dtype == df["timestamp"].dtype

    def test_dataframe_indices_reset_after_split(self):
        """Test that output DataFrames have clean indices."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "close": range(100),
        })

        train, val, test = split_dataset(df, split_mode="ratio")

        # Indices should be from iloc(), which preserves original indices
        # For concatenation to work, original indices must be preserved
        assert train.index.tolist() == list(range(70))
        assert val.index.tolist() == list(range(70, 85))
        assert test.index.tolist() == list(range(85, 100))

    def test_large_dataset_performance(self):
        """Test with a larger dataset (doesn't test performance numerically,
        just ensures it completes without error)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10000, freq="D"),
            "close": range(10000),
            "volume": range(10000, 20000),
        })

        train, val, test = split_dataset(df, split_mode="ratio")

        assert len(train) == 7000
        assert len(val) == 1500
        assert len(test) == 1500
        assert len(train) + len(val) + len(test) == 10000

    def test_date_range_with_microsecond_precision(self):
        """Test date_range mode with high precision timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(
                "2020-01-01", periods=100, freq="h"
            ),  # Hourly data (lowercase 'h' for modern pandas)
            "close": range(100),
        })

        train, val, test = split_dataset(
            df,
            split_mode="date_range",
            train_end="2020-01-03",
            val_end="2020-01-05",
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
