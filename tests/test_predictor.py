# tests/test_predictor.py

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.inference.predictor import Predictor
from src.models.gru import GRUConfig, GRUModel
from src.models.lstm import LSTMConfig, LSTMModel


def _history(rows: int = 100) -> pd.DataFrame:
    close = np.linspace(20.0, 30.0, rows)
    return pd.DataFrame(
        {
            "symbol": "INTC",
            "timestamp": pd.date_range("2024-01-01", periods=rows),
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000, 2000, rows),
        }
    )


@pytest.mark.parametrize(
    ("model_name", "model_config", "model_class"),
    [
        ("lstm", {"lookback": 10, "hidden_size": 8, "dropout": 0.0}, LSTMModel),
        ("gru", {"lookback": 10, "hidden_size": 8, "dropout": 0.0}, GRUModel),
    ],
)
def test_predictor_returns_finite_float_for_recurrent_checkpoint(
    tmp_path, model_name, model_config, model_class
):
    feature_names = ["return_1d"]
    model = model_class(
        (LSTMConfig if model_name == "lstm" else GRUConfig)(
            num_features=1, **model_config
        )
    )
    checkpoint_path = tmp_path / f"{model_name}.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    metadata_path = tmp_path / f"{model_name}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "symbol": "INTC",
                "model_name": model_name,
                "lookback": 10,
                "horizon": 1,
                "feature_names": feature_names,
                "target": "close",
                "scaler": {
                    "feature_names": feature_names,
                    "mean": [0.0],
                    "scale": [1.0],
                },
                "model_config": model_config,
            }
        )
    )

    result = Predictor(checkpoint_path, metadata_path).predict_next(_history())

    assert isinstance(result["predicted_price"], float)
    assert np.isfinite(result["predicted_price"])


def test_predictor_output_conversion_returns_list_for_multiple_predictions():
    output = torch.tensor([[1.0], [2.0]])

    assert Predictor._convert_torch_output(output) == [1.0, 2.0]
