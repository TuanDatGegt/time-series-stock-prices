## src/models/baseline.py
"""
Phase 10 - Baseline models.

Provides four baseline predictors that later deep-learning models
(LSTM, GRU) must be compared against:

    Naive -> MovingAverage -> LinearRegression -> XGBoost -> LSTM -> GRU

All four share the same fit/predict interface (BaselineModel) so they
can be trained, evaluated, and compared interchangeably by the same
harness. Naive and MovingAverage don't "learn" parameters in the ML
sense, but they still implement fit() (no-op or trivial) to satisfy
the shared interface.

Target definition (applies to ALL four models, for a fair comparison):
    y(t) = close(t + 1)   -- next-day closing price

Feature/target leakage rule:
    Only rows/columns available at time t may be used to predict
    close(t + 1). The "close" column at time t itself is a valid
    feature (it is known at prediction time); no column may be derived
    from t+1 or later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "xgboost is required for the XGBoost baseline. "
        "Install it with `pip install xgboost --break-system-packages`."
    ) from exc


TARGET_COL = "target_close_next"
CLOSE_COL = "close"


def make_target(df: pd.DataFrame, close_col: str = CLOSE_COL) -> pd.DataFrame:
    """
    Adds the next-day close target column to df.

    This is intentionally centralized here (not duplicated per model)
    so every baseline and every future model (LSTM/GRU) computes the
    target the exact same way -- one source of truth, avoids subtle
    mismatches between models being "compared" on different targets.

    The last row will have NaN target (no next day exists yet) and
    must be dropped by the caller before fitting/scoring.
    """
    out = df.copy()
    out[TARGET_COL] = out[close_col].shift(-1)
    return out


class BaselineModel(ABC):
    """Shared interface for all Phase 10 baseline predictors."""

    name: str

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaselineModel":
        """Fit the model on the training DataFrame (already sorted by
        timestamp, already has TARGET_COL via make_target, last-row
        NaN target already dropped)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return next-day close predictions, one per row of df."""
        raise NotImplementedError


@dataclass
class NaiveBaseline(BaselineModel):
    """prediction(t+1) = close(t)"""

    close_col: str = CLOSE_COL
    name: str = field(default="naive", init=False)

    def fit(self, train_df: pd.DataFrame) -> "NaiveBaseline":
        # No parameters to learn.
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.close_col].to_numpy()


@dataclass
class MovingAverageBaseline(BaselineModel):
    """prediction(t+1) = mean(last N closes), N = window"""

    window: int = 5
    close_col: str = CLOSE_COL
    name: str = field(default="moving_average", init=False)

    def fit(self, train_df: pd.DataFrame) -> "MovingAverageBaseline":
        if self.window < 1:
            raise ValueError("window must be >= 1")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (
            df[self.close_col]
            .rolling(window=self.window, min_periods=1)
            .mean()
            .to_numpy()
        )


class TabularBaseline(BaselineModel):
    """
    Shared logic for the two ML-based baselines (LinearRegression,
    XGBoost): both consume the same tabular feature matrix and only
    differ in which sklearn/xgboost estimator does the fitting.
    """

    def __init__(self, feature_cols: list[str]):
        if not feature_cols:
            raise ValueError("feature_cols must not be empty")
        if TARGET_COL in feature_cols:
            raise ValueError(
                f"{TARGET_COL!r} must not appear in feature_cols -- "
                "this would leak the target into the features."
            )
        self.feature_cols = feature_cols
        self._estimator = self._build_estimator()

    def _build_estimator(self):  # overridden by subclasses
        raise NotImplementedError

    def fit(self, train_df: pd.DataFrame) -> "TabularBaseline":
        missing = set(self.feature_cols) - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df is missing feature columns: {missing}")
        if TARGET_COL not in train_df.columns:
            raise ValueError(
                f"train_df must contain {TARGET_COL!r}; call make_target() first"
            )
        X = train_df[self.feature_cols].to_numpy()
        y = train_df[TARGET_COL].to_numpy()
        self._estimator.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].to_numpy()
        return self._estimator.predict(X)


class LinearRegressionBaseline(TabularBaseline):
    name = "linear_regression"

    def _build_estimator(self):
        return SklearnLinearRegression()


class XGBoostBaseline(TabularBaseline):
    name = "xgboost"

    def __init__(self, feature_cols: list[str], **xgb_kwargs):
        self._xgb_kwargs = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            **xgb_kwargs,
        }
        super().__init__(feature_cols)

    def _build_estimator(self):
        return XGBRegressor(**self._xgb_kwargs)


def build_all_baselines(
    feature_cols: list[str], ma_window: int = 5
) -> list[BaselineModel]:
    """Convenience factory returning all four baselines, ready to fit."""
    return [
        NaiveBaseline(),
        MovingAverageBaseline(window=ma_window),
        LinearRegressionBaseline(feature_cols=feature_cols),
        XGBoostBaseline(feature_cols=feature_cols),
    ]
