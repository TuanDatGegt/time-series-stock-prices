"""
Shared evaluation metrics for Phase 10 baselines and later phases
(Phase 15 - Evaluation will reuse these same functions so LSTM/GRU are
compared against baselines using an identical metric definition).
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, in percent. Rows where y_true == 0
    are excluded to avoid division by zero (shouldn't happen for stock
    close prices, but guarded defensively)."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(
    prev_close: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Fraction of rows where the model correctly predicted the DIRECTION
    of the next-day move (up/down/flat vs prev_close), not just the
    magnitude. This matters more than RMSE for trading-signal use
    cases and is easy to accidentally omit -- included explicitly here.
    """
    true_dir = np.sign(y_true - prev_close)
    pred_dir = np.sign(y_pred - prev_close)
    return float(np.mean(true_dir == pred_dir))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination (R^2). Added in Phase 15 alongside the
    Phase 10 metrics (rmse/mae/mape/directional_accuracy) so the full
    comparison table (MAE, RMSE, MAPE, R2, Direction Accuracy) can be
    built from one shared metrics module -- no metric is redefined
    per-phase.
 
    R^2 = 1 - (SS_res / SS_tot)
 
    Returns NaN if y_true is constant (SS_tot == 0), since R^2 is
    undefined in that degenerate case -- guarded rather than raising,
    consistent with how mape() already handles its own edge case.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - (ss_res / ss_tot))


def evaluate_predictions(
    prev_close: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> dict:
    """Returns one row (as a dict) of {model_name, metric_name: value, ...}
    suitable for collecting into a comparison table across all baselines
    (and, later, LSTM/GRU)."""
    return {
        "model_name": model_name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
        "directional_accuracy": directional_accuracy(prev_close, y_true, y_pred),
    }
