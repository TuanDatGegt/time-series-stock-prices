from __future__ import annotations

from typing import Any

import numpy as np

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
