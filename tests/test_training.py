import numpy as np
import pandas as pd
import pytest

from src.models.factory import create_model
from src.preprocessing.scaler import FeatureScaler
from src.preprocessing.sequence import build_sequences
from src.preprocessing.splitter import split_dataset
from src.training.train import run_training
from src.utils.config import load_config


def _market_data(rows=180):
    close = np.linspace(20.0, 40.0, rows)
    return pd.DataFrame(
        {
            "symbol": "INTC",
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000, 2000, rows),
        }
    )


def test_config_and_scaler_preserve_train_state():
    config = load_config()
    assert config["training"]["target_column"] == "close"

    scaler = FeatureScaler().fit(pd.DataFrame({"feature": [0.0, 2.0]}))
    transformed = scaler.transform(pd.DataFrame({"feature": [1.0, 3.0]}))
    np.testing.assert_allclose(transformed["feature"], [0.0, 2.0])
    assert scaler.state_dict()["feature_names"] == ["feature"]

    with pytest.raises(ValueError, match="do not match"):
        scaler.transform(pd.DataFrame({"other": [1.0]}))


def test_factory_supports_recurrent_models():
    lstm, lstm_config = create_model("lstm", 2, {"hidden_size": 4, "epochs": 1})
    gru, gru_config = create_model("gru", 2, {"hidden_size": 4, "epochs": 1})
    assert lstm.config.num_features == 2
    assert gru.config.num_features == 2
    assert lstm_config.epochs == 1
    assert gru_config.epochs == 1


def test_factory_supports_existing_xgboost_contract():
    model, model_config = create_model(
        "xgboost", 2, {"n_estimators": 2}, feature_cols=["a", "b"]
    )
    assert model.name == "xgboost"
    assert model.feature_cols == ["a", "b"]
    assert model_config["n_estimators"] == 2


def test_cli_parser_requires_symbol_and_model():
    from scripts.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    assert build_parser().parse_args(["--symbol", "INTC", "--model", "gru"]).model == "gru"


def test_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("unknown", 2)


def test_split_local_sequences_do_not_cross_boundaries():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=30),
            "close": np.arange(30, dtype=float),
            "feature": np.arange(30, dtype=float),
        }
    )
    train, validation, test = split_dataset(frame, train_ratio=0.6, val_ratio=0.2)
    train_X, train_y = build_sequences(train, lookback=3, feature_cols=["feature"])
    validation_X, validation_y = build_sequences(
        validation, lookback=3, feature_cols=["feature"]
    )
    assert train_X[0, 0, 0] == 0
    assert validation_X[0, 0, 0] == validation["feature"].iloc[0]
    assert train_y[-1, 0] == train["close"].iloc[-1]
    assert validation_y[0, 0] == validation["close"].iloc[3]
    assert test["timestamp"].min() > validation["timestamp"].max()


def test_tiny_training_creates_checkpoint_and_registration(tmp_path):
    config = load_config()
    config["training"].update(
        {
            "feature_columns": ["sma_5", "ema_12"],
            "lookback": 5,
            "horizon": 1,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        }
    )
    config["models"]["lstm"].update(
        {
            "lookback": 5,
            "hidden_size": 4,
            "num_layers": 1,
            "epochs": 1,
            "batch_size": 8,
            "patience": 10,
        }
    )
    config["device"] = "cpu"

    result = run_training(
        "INTC",
        "lstm",
        config=config,
        data_loader=lambda symbol: _market_data(),
        checkpoint_dir=tmp_path / "checkpoints",
        metadata_dir=tmp_path / "metadata",
    )

    assert result.validation_metrics["model_name"] == "lstm"
    assert "mae" in result.test_metrics
    assert result.history[0]["epoch"] == 1.0
    assert result.checkpoint_path.endswith("lstm_best.pt")
    assert len(result.history) <= config["models"]["lstm"]["epochs"]
    assert (tmp_path / "checkpoints" / "INTC" / "lstm_best.pt").exists()
    assert (tmp_path / "checkpoints" / "INTC" / "lstm.pt").exists()
    assert (tmp_path / "metadata" / "INTC" / "lstm.json").exists()


def test_early_stopping_breaks_training_loop(monkeypatch):
    import torch

    from src.training import train as training_module

    class StopAfterFirstValidation:
        def __init__(self, patience, min_delta):
            self.calls = 0

        def step(self, val_loss, epoch):
            self.calls += 1
            return type("Result", (), {"improved": True, "should_stop": True})()

    monkeypatch.setattr(training_module, "EarlyStopping", StopAfterFirstValidation)
    model = torch.nn.Linear(1, 1)
    data = np.ones((4, 1), dtype=np.float32)
    targets = np.ones((4, 1), dtype=np.float32)

    history, _, _ = training_module._train_torch(
        model,
        data,
        targets,
        data,
        targets,
        type("Config", (), {"learning_rate": 0.01, "batch_size": 2, "epochs": 5})(),
        "cpu",
        patience=0,
        min_delta=0.0,
    )

    assert len(history) == 1
