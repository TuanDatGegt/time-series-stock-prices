from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"Configuration file not found: {path}")
	with path.open("r", encoding="utf-8") as handle:
		value = yaml.safe_load(handle) or {}
	if not isinstance(value, dict):
		raise ValueError(f"Configuration file must contain a mapping: {path}")
	return value


def load_config(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
	"""Load and validate the small set of configuration used by Phase 13."""
	directory = Path(config_dir)
	config = _load_yaml(directory / "config.yaml")
	data = _load_yaml(directory / "data.yaml")
	model = _load_yaml(directory / "model.yaml")

	result = {**config, "data": data, "models": model}
	result.setdefault("data", {})
	result.setdefault("training", {})
	result.setdefault("paths", {})
	result.setdefault("logging", {})

	training = result["training"]
	required = ("feature_columns", "target_column", "lookback", "horizon")
	missing = [key for key in required if key not in training]
	if missing:
		raise ValueError(f"Missing training configuration values: {missing}")
	if not isinstance(training["feature_columns"], list) or not training["feature_columns"]:
		raise ValueError("training.feature_columns must be a non-empty list")
	for key in ("lookback", "horizon"):
		if not isinstance(training[key], int) or training[key] < 1:
			raise ValueError(f"training.{key} must be a positive integer")

	return result
