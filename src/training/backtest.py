## src/training/backtest.py
"""
Module: src/training/backtest.py
Description: Walk-Forward Rolling Window Backtesting Engine.
How it works:
    Performs walk-forward (expanding/rolling window) backtesting across historical time slices.
    In each fold, it fits preprocessing scalers strictly on the training window, builds sequences/features,
    trains the candidate model, predicts the out-of-sample test horizon, and computes evaluation metrics
    via `src.training.evaluate.evaluate_arrays`. Finally, it aggregates fold results to measure model stability over time.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.preprocessing.scaler import FeatureScaler
from src.preprocessing.sequence import build_sequences
from src.training.evaluate import evaluate_arrays


@dataclass
class BacktestFoldResult:
    """
    Stores execution results and evaluation metrics for a single backtest fold.
    """

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: Dict[str, float]


class WalkForwardBacktester:
    """
    Executes walk-forward backtesting for time-series forecasting models.
    """

    def __init__(
        self,
        lookback: int = 60,
        horizon: int = 1,
        n_folds: int = 3,
        initial_train_ratio: float = 0.6,
        feature_cols: List[str] | None = None,
        target_col: str = "close",
    ):
        """
        Initialize backtesting parameters and feature/target definitions.
        """
        self.lookback = lookback
        self.horizon = horizon
        self.n_folds = n_folds
        self.initial_train_ratio = initial_train_ratio
        self.feature_cols = feature_cols
        self.target_col = target_col

    def generate_folds(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split a chronologically sorted DataFrame into expanding training and testing window folds.
        """
        n_rows = len(df)
        min_train_size = int(n_rows * self.initial_train_ratio)
        remaining_rows = n_rows - min_train_size

        if remaining_rows < self.n_folds:
            raise ValueError(
                f"Insufficient data for {self.n_folds} folds: {remaining_rows} rows remaining."
            )

        fold_size = remaining_rows // self.n_folds
        folds = []

        for i in range(self.n_folds):
            train_end_idx = min_train_size + (i * fold_size)
            test_end_idx = (
                min_train_size + ((i + 1) * fold_size)
                if i < self.n_folds - 1
                else n_rows
            )

            train_split = df.iloc[:train_end_idx].copy()
            test_split = df.iloc[train_end_idx:test_end_idx].copy()
            folds.append((train_split, test_split))

        return folds

    def run_backtest(
        self,
        df: pd.DataFrame,
        model_trainer_fn: Any,
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, List[BacktestFoldResult]]:
        """
        Execute the walk-forward backtesting pipeline across all folds.

        Args:
            df (pd.DataFrame): Input market dataset sorted by timestamp.
            model_trainer_fn: Function or factory returning trained model predictions given (train_df, test_df).
            timestamp_col (str): Timestamp column name.

        Returns:
            Tuple[pd.DataFrame, List[BacktestFoldResult]]: Summary metrics DataFrame and detailed fold results.
        """
        folds = self.generate_folds(df)
        fold_results: List[BacktestFoldResult] = []
        metrics_list: List[Dict[str, Any]] = []

        for idx, (train_df, test_df) in enumerate(folds):
            train_start = str(train_df[timestamp_col].min())
            train_end = str(train_df[timestamp_col].max())
            test_start = str(test_df[timestamp_col].min())
            test_end = str(test_df[timestamp_col].max())

            # Fit scaler strictly on training split to prevent feature leakage
            scaler = FeatureScaler().fit(train_df[self.feature_cols])
            scaled_train = train_df.copy()
            scaled_test = test_df.copy()

            scaled_train[self.feature_cols] = scaler.transform(
                train_df[self.feature_cols]
            ).to_numpy()
            scaled_test[self.feature_cols] = scaler.transform(
                test_df[self.feature_cols]
            ).to_numpy()

            # Execute model training and inference for current fold
            model_name, y_true, y_pred, prev_close = model_trainer_fn(
                scaled_train,
                scaled_test,
                self.lookback,
                self.horizon,
                self.feature_cols,
                self.target_col,
            )

            # Evaluate metrics for current fold
            fold_metrics = evaluate_arrays(
                prev_close, y_true, y_pred, model_name=f"{model_name}_fold_{idx+1}"
            )

            fold_res = BacktestFoldResult(
                fold_index=idx + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                metrics=fold_metrics,
            )
            fold_results.append(fold_res)

            metrics_record = {
                "fold": idx + 1,
                "train_range": f"{train_start} to {train_end}",
                "test_range": f"{test_start} to {test_end}",
                **fold_metrics,
            }
            metrics_list.append(metrics_record)

        summary_df = pd.DataFrame(metrics_list)
        return summary_df, fold_results
