#tests/test_evaluate.py

import numpy as np
import pandas as pd
import pytest

from src.training.evaluate import (
	evaluate_all_models,
	evaluate_arrays,
	format_comparison_table,
	identify_best_model,
)


@pytest.fixture
def synthetic_predictions():
	"""4 models with hand-crafted errors so ranking is predictable:
	naive is worst (large error), lstm is best (near-perfect)."""
	y_true = np.array([10.0, 11.0, 9.0, 12.0, 13.0])
	prev_close = np.array([9.5, 10.0, 11.0, 9.0, 12.0])

	predictions = {
		"naive": np.array([9.5, 10.0, 11.0, 9.0, 12.0]),   # large errors
		"xgboost": np.array([9.8, 10.7, 9.3, 11.7, 12.7]),  # medium errors
		"lstm": np.array([10.0, 11.0, 9.0, 12.0, 13.0]),    # perfect
		"gru": np.array([9.9, 11.1, 8.9, 12.1, 12.9]),      # tiny errors
	}
	return predictions, y_true, prev_close


# ---------------------------------------------------------------------
# evaluate_arrays (existing Phase 13 function -- regression coverage,
# since it previously had no dedicated tests)
# ---------------------------------------------------------------------

def test_evaluate_arrays_returns_all_metric_keys():
	result = evaluate_arrays(
		prev_close=np.array([9.0, 10.0]),
		y_true=np.array([10.0, 11.0]),
		y_pred=np.array([9.8, 10.9]),
		model_name="lstm",
	)
	assert result["model_name"] == "lstm"
	for key in ("mae", "rmse", "mape", "r_squared", "directional_accuracy"):
		assert key in result


def test_evaluate_arrays_rejects_unequal_lengths():
	with pytest.raises(ValueError, match="equal lengths"):
		evaluate_arrays(
			prev_close=np.array([9.0, 10.0]),
			y_true=np.array([10.0]),
			y_pred=np.array([9.8, 10.9]),
			model_name="lstm",
		)


def test_evaluate_arrays_rejects_empty_input():
	with pytest.raises(ValueError, match="empty"):
		evaluate_arrays(
			prev_close=np.array([]),
			y_true=np.array([]),
			y_pred=np.array([]),
			model_name="lstm",
		)


def test_evaluate_arrays_rejects_non_finite_values():
	with pytest.raises(ValueError, match="finite"):
		evaluate_arrays(
			prev_close=np.array([9.0, 10.0]),
			y_true=np.array([10.0, np.nan]),
			y_pred=np.array([9.8, 10.9]),
			model_name="lstm",
		)


# ---------------------------------------------------------------------
# evaluate_all_models
# ---------------------------------------------------------------------

def test_evaluate_all_models_returns_one_row_per_model(synthetic_predictions):
	predictions, y_true, prev_close = synthetic_predictions
	df = evaluate_all_models(predictions, y_true, prev_close)
	assert len(df) == 4
	assert set(df["model_name"]) == {"naive", "xgboost", "lstm", "gru"}


def test_evaluate_all_models_sorted_best_to_worst_by_rmse(synthetic_predictions):
	predictions, y_true, prev_close = synthetic_predictions
	df = evaluate_all_models(predictions, y_true, prev_close)
	assert df.iloc[0]["model_name"] == "lstm"
	assert df.iloc[0]["rmse"] == pytest.approx(0.0)
	assert df.iloc[-1]["model_name"] == "naive"


def test_evaluate_all_models_rejects_empty_predictions():
	with pytest.raises(ValueError, match="at least one model"):
		evaluate_all_models({}, np.array([1.0]), np.array([1.0]))


def test_evaluate_all_models_propagates_shape_mismatch_from_evaluate_arrays():
	"""Confirms evaluate_all_models does NOT duplicate its own shape
	check -- it must surface evaluate_arrays' own ValueError instead."""
	y_true = np.array([1.0, 2.0, 3.0])
	prev_close = np.array([1.0, 2.0, 3.0])
	predictions = {"bad_model": np.array([1.0, 2.0])}  # wrong length
	with pytest.raises(ValueError, match="equal lengths"):
		evaluate_all_models(predictions, y_true, prev_close)


# ---------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------

def test_format_comparison_table_includes_all_model_names(synthetic_predictions):
	predictions, y_true, prev_close = synthetic_predictions
	df = evaluate_all_models(predictions, y_true, prev_close)
	table_str = format_comparison_table(df)
	for name in ("naive", "xgboost", "lstm", "gru"):
		assert name in table_str


def test_format_comparison_table_has_header_and_separator(synthetic_predictions):
	predictions, y_true, prev_close = synthetic_predictions
	df = evaluate_all_models(predictions, y_true, prev_close)
	table_str = format_comparison_table(df)
	lines = table_str.splitlines()
	assert "Model" in lines[0]
	assert set(lines[1]) == {"-"}


# ---------------------------------------------------------------------
# identify_best_model
# ---------------------------------------------------------------------

def test_identify_best_model_by_rmse(synthetic_predictions):
	predictions, y_true, prev_close = synthetic_predictions
	df = evaluate_all_models(predictions, y_true, prev_close)
	assert identify_best_model(df, metric="rmse") == "lstm"


def test_identify_best_model_does_not_hard_code_lstm_preference():
	"""Directly tests the spec's warning: if a baseline genuinely beats
	LSTM, identify_best_model must say so, not silently prefer LSTM."""
	y_true = np.array([10.0, 11.0, 9.0])
	prev_close = np.array([9.5, 10.0, 11.0])
	predictions = {
		"naive": np.array([10.0, 11.0, 9.0]),   # perfect
		"lstm": np.array([5.0, 20.0, 1.0]),     # terrible
	}
	df = evaluate_all_models(predictions, y_true, prev_close)
	assert identify_best_model(df, metric="rmse") == "naive"


def test_identify_best_model_rejects_unknown_metric(synthetic_predictions):
	predictions, y_true, prev_close = synthetic_predictions
	df = evaluate_all_models(predictions, y_true, prev_close)
	with pytest.raises(ValueError):
		identify_best_model(df, metric="not_a_real_metric")