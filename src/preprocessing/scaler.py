## src/preprocessing/scaler.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureScaler:
    """Feature scaler that records the exact ordered columns it fitted on."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    feature_names: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> "FeatureScaler":
        self._validate_frame(features)
        self.feature_names = list(features.columns)
        values = features.to_numpy(dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        self._validate_frame(features)
        if self.mean is None or self.scale is None or self.feature_names is None:
            raise ValueError("FeatureScaler must be fitted before transform")
        if list(features.columns) != self.feature_names:
            raise ValueError(
                f"Feature columns do not match fitted columns: "
                f"expected {self.feature_names}, got {list(features.columns)}"
            )
        values = (features.to_numpy(dtype=np.float64) - self.mean) / self.scale
        return pd.DataFrame(values, index=features.index, columns=self.feature_names)

    def state_dict(self) -> dict:
        if self.mean is None or self.scale is None or self.feature_names is None:
            raise ValueError("Cannot serialize an unfitted FeatureScaler")
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @staticmethod
    def _validate_frame(features: pd.DataFrame) -> None:
        if features is None or features.empty:
            raise ValueError("Features cannot be empty")
        if not all(
            pd.api.types.is_numeric_dtype(features[column]) for column in features
        ):
            raise ValueError("All features must be numeric")
        if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
            raise ValueError("Features must contain only finite values")
