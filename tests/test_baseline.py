# tests/test_baseline.py
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    CLOSE_COL,
    TARGET_COL,
    LinearRegressionBaseline,
    MovingAverageBaseline,
    NaiveBaseline,
    XGBoostBaseline,
    build_all_baselines,
    make_target,
)
from src.models.metrics import (
    directional_accuracy,
    evaluate_predictions,
    mae,
    mape,
    rmse,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_df():
    """5 rows of hand-crafted close prices so expected outputs can be
    verified by hand, not just by re-running the same formula."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "close": [10.0, 12.0, 11.0, 13.0, 15.0],
            "volume": [100, 110, 105, 120, 130],
        }
    )


# ---------------------------------------------------------------------
# make_target
# ---------------------------------------------------------------------

def test_make_target_shifts_close_by_one(tiny_df):
    out = make_target(tiny_df)
    expected = [12.0, 11.0, 13.0, 15.0, np.nan]
    assert out[TARGET_COL].tolist()[:-1] == expected[:-1]
    assert np.isnan(out[TARGET_COL].iloc[-1])


# ---------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------

def test_naive_predicts_previous_close(tiny_df):
    model = NaiveBaseline().fit(tiny_df)
    preds = model.predict(tiny_df)
    # Naive prediction(t+1) = close(t) -> predict() just returns close as-is,
    # the "shift" semantics live in how predictions are compared to target.
    assert preds.tolist() == [10.0, 12.0, 11.0, 13.0, 15.0]


# ---------------------------------------------------------------------
# Moving average baseline
# ---------------------------------------------------------------------

def test_moving_average_window_3(tiny_df):
    model = MovingAverageBaseline(window=3).fit(tiny_df)
    preds = model.predict(tiny_df)
    # rolling mean, min_periods=1:
    # [10, (10+12)/2, (10+12+11)/3, (12+11+13)/3, (11+13+15)/3]
    expected = [10.0, 11.0, 11.0, 12.0, 13.0]
    assert preds == pytest.approx(expected)


def test_moving_average_rejects_invalid_window():
    with pytest.raises(ValueError):
        MovingAverageBaseline(window=0).fit(pd.DataFrame({"close": [1.0]}))


# ---------------------------------------------------------------------
# Tabular baselines (LinearRegression, XGBoost)
# ---------------------------------------------------------------------

def test_tabular_baseline_rejects_target_in_features(tiny_df):
    df = make_target(tiny_df).dropna(subset=[TARGET_COL])
    with pytest.raises(ValueError):
        LinearRegressionBaseline(feature_cols=[CLOSE_COL, TARGET_COL])


def test_linear_regression_fits_and_predicts(tiny_df):
    df = make_target(tiny_df).dropna(subset=[TARGET_COL])
    model = LinearRegressionBaseline(feature_cols=["close", "volume"])
    model.fit(df)
    preds = model.predict(df)
    assert preds.shape[0] == len(df)
    assert np.all(np.isfinite(preds))


def test_xgboost_fits_and_predicts(tiny_df):
    df = make_target(tiny_df).dropna(subset=[TARGET_COL])
    model = XGBoostBaseline(feature_cols=["close", "volume"], n_estimators=5)
    model.fit(df)
    preds = model.predict(df)
    assert preds.shape[0] == len(df)
    assert np.all(np.isfinite(preds))


def test_tabular_baseline_raises_on_missing_feature_column(tiny_df):
    df = make_target(tiny_df).dropna(subset=[TARGET_COL])
    model = LinearRegressionBaseline(feature_cols=["close", "does_not_exist"])
    with pytest.raises(ValueError):
        model.fit(df)


# ---------------------------------------------------------------------
# build_all_baselines
# ---------------------------------------------------------------------

def test_build_all_baselines_returns_four_models():
    models = build_all_baselines(feature_cols=["close", "volume"])
    assert len(models) == 4
    names = {m.name for m in models}
    assert names == {"naive", "moving_average", "linear_regression", "xgboost"}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def test_rmse_mae_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_directional_accuracy_all_correct():
    prev_close = np.array([10.0, 10.0])
    y_true = np.array([12.0, 8.0])  # up, down
    y_pred = np.array([11.0, 9.0])  # predicted up, predicted down -> correct direction
    assert directional_accuracy(prev_close, y_true, y_pred) == 1.0


def test_directional_accuracy_all_wrong():
    prev_close = np.array([10.0, 10.0])
    y_true = np.array([12.0, 8.0])  # up, down
    y_pred = np.array([9.0, 11.0])  # predicted down, predicted up -> both wrong
    assert directional_accuracy(prev_close, y_true, y_pred) == 0.0


def test_evaluate_predictions_returns_all_metric_keys():
    prev_close = np.array([10.0, 10.0])
    y_true = np.array([12.0, 8.0])
    y_pred = np.array([11.0, 9.0])
    result = evaluate_predictions(prev_close, y_true, y_pred, model_name="naive")
    assert result["model_name"] == "naive"
    for key in ("rmse", "mae", "mape", "directional_accuracy"):
        assert key in result