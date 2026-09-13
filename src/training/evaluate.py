## src/training/evaluate.py

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from src.models.metrics import evaluate_predictions


def evaluate_arrays(
    prev_close: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate aligned predictions using the shared Phase 10 metrics."""
    previous = np.asarray(prev_close, dtype=np.float64).reshape(-1)
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predictions = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if not (len(previous) == len(truth) == len(predictions)):
        raise ValueError("prev_close, y_true, and y_pred must have equal lengths")
    if not len(truth):
        raise ValueError("Cannot evaluate an empty prediction set")
    if not np.isfinite(np.concatenate([previous, truth, predictions])).all():
        raise ValueError("Evaluation arrays must contain only finite values")
    return evaluate_predictions(previous, truth, predictions, model_name=model_name)


def predict_torch(model, features: np.ndarray, device: str) -> np.ndarray:
    """Run inference for a recurrent model without changing model state."""
    import torch

    model.eval()
    inputs = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return model(inputs).detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------
# Phase 15 - Comparison table across all models.
#
# Builds on evaluate_arrays() above (does not duplicate its validation
# logic) to produce the multi-model table shown in the spec:
#
#     Model      MAE    RMSE    MAPE    Direction
#     ------------------------------------------------
#     Naive      1.92   2.75    7.8%    50.1%
#     XGBoost    1.55   2.21    6.4%    57.3%
#     LSTM       1.41   2.08    5.9%    60.2%
#     GRU        1.46   2.13    6.1%    59.7%
# ---------------------------------------------------------------------


def evaluate_all_models(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    prev_close: np.ndarray,
) -> pd.DataFrame:
    """
    predictions: {model_name: y_pred_array}, e.g.
        {"naive": ..., "xgboost": ..., "lstm": ..., "gru": ...}
    y_true, prev_close: same for every model -- all models are scored
        on the identical test set, otherwise the comparison is meaningless.

    Each model's row is computed via evaluate_arrays(), so shape/finite
    validation happens in exactly one place (this function does not
    re-validate inputs itself -- it lets evaluate_arrays raise).

    Returns a DataFrame sorted best-to-worst by RMSE (lower is better),
    matching the spec's example table ordering.
    """
    if not predictions:
        raise ValueError("predictions must contain at least one model")

    rows = [
        evaluate_arrays(prev_close, y_true, y_pred, model_name=model_name)
        for model_name, y_pred in predictions.items()
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values("rmse", ascending=True).reset_index(drop=True)
    return df


def format_comparison_table(df: pd.DataFrame) -> str:
    """
    Renders the DataFrame as a human-readable table matching the
    spec's example format (percent signs on mape/directional_accuracy,
    fixed decimal places), suitable for printing to console or logging.
    """
    lines = [
        f"{'Model':<12}{'MAE':<8}{'RMSE':<8}{'MAPE':<9}{'R_Squared':<8}{'Direction':<10}"
    ]
    lines.append("-" * 55)
    for _, row in df.iterrows():
        lines.append(
            f"{row['model_name']:<12}"
            f"{row['mae']:<8.2f}"
            f"{row['rmse']:<8.2f}"
            f"{row['mape']:<8.1f}%"
            f"{row['r_squared']:<8.2f}"
            f"{row['directional_accuracy'] * 100:<9.1f}%"
        )
    return "\n".join(lines)


def identify_best_model(df: pd.DataFrame, metric: str = "rmse") -> str:
    """
    Returns the model_name with the best (lowest, for error metrics;
    highest, for r2/directional_accuracy) score on the given metric.

    Per the spec's warning in Phase 10 ("if LSTM isn't better than
    baseline, investigate why -- don't default to picking it anyway"),
    this is a plain lookup, not a hard-coded preference for LSTM/GRU --
    a baseline CAN legitimately win here, and callers must respect that
    rather than overriding it.
    """
    higher_is_better = metric in ("r_squared", "directional_accuracy")
    if metric not in df.columns:
        raise ValueError(
            f"Unknown metric {metric!r}; must be one of {list(df.columns)}"
        )

    idx = df[metric].idxmax() if higher_is_better else df[metric].idxmin()
    return str(df.loc[idx, "model_name"])
