## src/preprocessing/sequence.py
"""
Sliding-window sequence creation for time-series forecasting.

This module creates input sequences (X) and targets (y) from a time-series DataFrame.
Each sequence is a sliding window of historical data, with the target being a future value.

CRITICAL CONSTRAINTS:
- Called AFTER split_dataset() and AFTER scaling
- Each train/val/test split processed independently
- NO future data leakage into sequences
- NO shuffling or data manipulation
- Preserves chronological order strictly
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    lookback: int = 60,
    horizon: int = 1,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "close",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences from a time-series DataFrame.

    This function generates input sequences (X) and corresponding targets (y)
    using a sliding window approach. It operates on a single time-series split
    (train, validation, or test) without cross-split contamination.

    **CRITICAL**: Call this AFTER split_dataset() and scaling.
    Each split must be processed independently to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series data sorted by timestamp (ascending order).
        Must contain feature_cols and target_col.
    lookback : int, optional
        Number of historical observations (window size). Default: 60.
    horizon : int, optional
        Steps ahead to predict. Default: 1 (one-step ahead).
    feature_cols : list[str] | None, optional
        Column names to use as features. If None, uses all numeric columns
        except timestamp, symbol, OHLCV (open, high, low, close, volume).
    target_col : str, optional
        Column name for the target variable. Default: "close".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X : ndarray of shape (n_samples, lookback, n_features), dtype float32
            Input sequences.
        y : ndarray of shape (n_samples, 1), dtype float32
            Target values aligned with X (shifted forward by horizon steps).

    Raises
    ------
    ValueError
        - DataFrame is empty
        - lookback or horizon not a positive integer
        - Insufficient rows (len(df) < lookback + horizon)
        - target_col not in DataFrame columns
        - feature_cols contains non-existent columns
        - feature/target data contains non-numeric values that cannot be converted

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.date_range('2020-01-01', periods=100),
    ...     'close': np.arange(100, dtype=float),
    ...     'sma_20': np.arange(100, dtype=float) * 0.99,
    ... })
    >>> X, y = build_sequences(df, lookback=10, horizon=1)
    >>> X.shape
    (89, 10, 2)
    >>> y.shape
    (89, 1)

    Notes
    -----
    - Sequence alignment: X[i] contains rows [i, i+1, ..., i+lookback-1]
      and y[i] contains row [i+lookback+horizon-1]
    - Number of sequences: len(df) - lookback - horizon + 1
    - NaN values are preserved in sequences; caller is responsible
      for imputation/removal before or after sequence creation
    - Features are not scaled; use already-scaled data or scale before calling
    """
    # ========================================================================
    # VALIDATION
    # ========================================================================

    if df is None or df.empty:
        raise ValueError("DataFrame cannot be empty.")

    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError(f"lookback must be a positive integer, got {lookback}")

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")

    # Check sufficient data
    min_rows = lookback + horizon
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient data: {len(df)} rows available, "
            f"but need at least {min_rows} (lookback={lookback} + horizon={horizon})"
        )

    # Check target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Determine feature columns
    if feature_cols is None:
        # Default: all numeric except timestamp, symbol, and OHLCV
        exclude_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

    # Validate feature columns exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_cols}")

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================

    # Extract feature data (numeric conversion with error checking)
    try:
        X_data = df[feature_cols].astype(np.float32).values
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert feature data to float32: {e}")

    # Extract target data
    try:
        y_data = df[target_col].astype(np.float32).values
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert target column '{target_col}' to float32: {e}")

    # ========================================================================
    # SEQUENCE CREATION (Sliding Window)
    # ========================================================================

    n_samples = len(df) - lookback - horizon + 1

    if n_samples <= 0:
        raise ValueError(
            f"No valid sequences can be created: "
            f"n_samples = len(df) - lookback - horizon + 1 = {len(df)} - {lookback} - {horizon} + 1 = {n_samples}"
        )

    # Pre-allocate arrays
    X = np.zeros((n_samples, lookback, X_data.shape[1]), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

    # Create sequences via sliding window
    for i in range(n_samples):
        # Input window: rows [i, i+1, ..., i+lookback-1]
        X[i] = X_data[i : i + lookback]
        # Target: row [i+lookback+horizon-1]
        y[i] = y_data[i + lookback + horizon - 1]

    return X, y
