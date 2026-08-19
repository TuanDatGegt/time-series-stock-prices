from __future__ import annotations

from typing import Any

from src.models.gru import GRUConfig, GRUModel
from src.models.lstm import LSTMConfig, LSTMModel


SUPPORTED_MODELS = ("lstm", "gru", "xgboost")


def create_model(
	model_name: str,
	num_features: int,
	model_config: dict[str, Any] | None = None,
	feature_cols: list[str] | None = None,
):
	"""Build a Phase 13 model using the existing model contracts."""
	name = model_name.lower()
	if name not in SUPPORTED_MODELS:
		raise ValueError(f"Unknown model '{model_name}'. Supported models: {SUPPORTED_MODELS}")
	raw = model_config or {}
	if name == "lstm":
		config = LSTMConfig.from_dict(num_features=num_features, raw=raw)
		return LSTMModel(config), config
	if name == "gru":
		config = GRUConfig.from_dict(num_features=num_features, raw=raw)
		return GRUModel(config), config
	if not feature_cols:
		raise ValueError("feature_cols are required for xgboost")
	try:
		from src.models.baseline import XGBoostBaseline
	except ImportError as exc:
		raise ImportError(
			"xgboost training requires the project's scikit-learn and "
			"xgboost dependencies"
		) from exc
	return XGBoostBaseline(feature_cols=feature_cols, **raw), raw
