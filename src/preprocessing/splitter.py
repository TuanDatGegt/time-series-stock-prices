"""
Chronological dataset splitter for time-series forecasting.

This module provides functions to split time-series data into train, validation,
and test sets while preserving temporal order. NO random shuffling is performed
to avoid data leakage (using future data to predict the past).
"""
from __future__ import annotations

from typing import Literal

import pandas as pd


def split_dataset(
    df: pd.DataFrame,
    split_mode: Literal["date_range", "ratio"] = "ratio",
    timestamp_col: str = "timestamp",
    train_end: str | None = None,
    val_end: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series dataset into train, validation, and test sets.

    CRITICAL: This function preserves chronological order. No random shuffling.
    This is essential for time-series data to avoid data leakage (using future
    information to make past predictions).

    Parameters
    ----------
    df : pd.DataFrame
        Input data sorted by timestamp_col (ascending). Must not be empty.
    split_mode : Literal["date_range", "ratio"]
        "date_range": Use train_end and val_end as explicit date boundaries.
        "ratio": Split using train_ratio and val_ratio proportions.
        Default: "ratio".
    timestamp_col : str
        Name of the timestamp column. Default: "timestamp".
    train_end : str | None
        End boundary for training set (inclusive). Used in "date_range" mode.
        Format: parseable by pd.to_datetime (e.g., "2023-12-31").
    val_end : str | None
        End boundary for validation set (inclusive). Used in "date_range" mode.
        Format: parseable by pd.to_datetime.
    train_ratio : float
        Proportion of data for training (0 < train_ratio < 1).
        Used in "ratio" mode. Default: 0.7.
    val_ratio : float
        Proportion of data for validation (0 < val_ratio < 1).
        Used in "ratio" mode. Default: 0.15.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) as non-overlapping DataFrames in chronological order.
        Concatenating all three (sorted by timestamp) reconstructs the original df.

    Raises
    ------
    ValueError
        If any input validation fails:
        - DataFrame is empty
        - timestamp_col not found in columns
        - DataFrame not sorted by timestamp_col (ascending)
        - Invalid ratios (must sum to < 1.0, must be > 0.0)
        - Invalid date_range parameters (train_end >= val_end, missing dates)
        - Date boundaries outside data range

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "timestamp": pd.date_range("2020-01-01", periods=100),
    ...     "close": range(100),
    ... })
    >>> train, val, test = split_dataset(df, train_ratio=0.6, val_ratio=0.2)
    >>> len(train), len(val), len(test)
    (60, 20, 20)

    >>> # Using explicit date boundaries:
    >>> train, val, test = split_dataset(
    ...     df,
    ...     split_mode="date_range",
    ...     train_end="2020-02-10",
    ...     val_end="2020-03-10",
    ... )
    """
    # ============================================================================
    # INPUT VALIDATION
    # ============================================================================

    # Check if dataframe is empty
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    # Check if timestamp column exists
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Check if df is sorted by timestamp_col (ascending).
    # NOTE: We do NOT silently re-sort. The caller is responsible for sorting.
    is_sorted = df[timestamp_col].is_monotonic_increasing
    if not is_sorted:
        raise ValueError(
            f"DataFrame must be sorted by '{timestamp_col}' in ascending order. "
            f"Caller is responsible for sorting."
        )

    # ============================================================================
    # SPLIT LOGIC
    # ============================================================================

    if split_mode == "date_range":
        return _split_by_date_range(df, timestamp_col, train_end, val_end)
    elif split_mode == "ratio":
        return _split_by_ratio(df, timestamp_col, train_ratio, val_ratio)
    else:
        raise ValueError(
            f"split_mode must be 'date_range' or 'ratio', got '{split_mode}'"
        )


def _split_by_ratio(
    df: pd.DataFrame,
    timestamp_col: str,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by proportional ratios without random shuffling.

    Computes row-index cutoffs based on ratios and splits sequentially.
    """
    # Validate ratios
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must sum to < 1.0. "
            f"Sum = {train_ratio + val_ratio}"
        )

    if train_ratio <= 0.0 or val_ratio <= 0.0:
        raise ValueError(
            f"train_ratio and val_ratio must be > 0.0. "
            f"Got train_ratio={train_ratio}, val_ratio={val_ratio}"
        )

    n_total = len(df)

    # Calculate cutoff indices (using int for truncation)
    train_size = int(n_total * train_ratio)
    val_size = int(n_total * val_ratio)

    # Edge case: ensure at least 1 row in each set if possible
    if train_size == 0:
        raise ValueError(
            f"train_ratio {train_ratio} results in 0 training samples "
            f"(df has {n_total} rows). Increase train_ratio."
        )

    if val_size == 0:
        raise ValueError(
            f"val_ratio {val_ratio} results in 0 validation samples "
            f"(df has {n_total} rows). Increase val_ratio."
        )

    # Split by indices (pure sequential, no shuffling)
    train = df.iloc[:train_size].copy()
    val = df.iloc[train_size : train_size + val_size].copy()
    test = df.iloc[train_size + val_size :].copy()

    return train, val, test


def _split_by_date_range(
    df: pd.DataFrame,
    timestamp_col: str,
    train_end: str | None,
    val_end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by explicit date boundaries (inclusive).

    Train: timestamp <= train_end
    Val:   train_end < timestamp <= val_end
    Test:  timestamp > val_end
    """
    # Validate date_range mode parameters
    if train_end is None or val_end is None:
        raise ValueError(
            "date_range mode requires both train_end and val_end parameters. "
            f"Got train_end={train_end}, val_end={val_end}"
        )

    # Parse dates
    try:
        train_end_ts = pd.to_datetime(train_end)
    except Exception as e:
        raise ValueError(f"Cannot parse train_end '{train_end}': {e}")

    try:
        val_end_ts = pd.to_datetime(val_end)
    except Exception as e:
        raise ValueError(f"Cannot parse val_end '{val_end}': {e}")

    # Validate date range
    if train_end_ts >= val_end_ts:
        raise ValueError(
            f"train_end must be < val_end. "
            f"Got train_end={train_end_ts}, val_end={val_end_ts}"
        )

    # Split by date boundaries
    train = df[df[timestamp_col] <= train_end_ts].copy()
    val = df[(df[timestamp_col] > train_end_ts) & (df[timestamp_col] <= val_end_ts)].copy()
    test = df[df[timestamp_col] > val_end_ts].copy()

    # Ensure no empty splits
    if train.empty:
        raise ValueError(
            f"No data found for train set with train_end='{train_end}'. "
            f"Min timestamp in df: {df[timestamp_col].min()}"
        )

    if val.empty:
        raise ValueError(
            f"No data found for validation set with train_end='{train_end}' "
            f"and val_end='{val_end}'."
        )

    if test.empty:
        raise ValueError(
            f"No data found for test set with val_end='{val_end}'. "
            f"Max timestamp in df: {df[timestamp_col].max()}"
        )

    return train, val, test
