# tests/test_gru.py
import pytest
import torch

from src.models.gru import GRUConfig, GRUModel


# ---------------------------------------------------------------------
# GRUConfig
# ---------------------------------------------------------------------

def test_config_defaults_are_lighter_than_lstm():
    config = GRUConfig(num_features=10)
    assert config.lookback == 60
    assert config.hidden_size == 64      # lighter than LSTM's 128
    assert config.num_layers == 1        # lighter than LSTM's 2
    assert config.dropout == 0.2
    assert config.learning_rate == 0.001
    assert config.batch_size == 64
    assert config.epochs == 50


def test_config_from_dict_overrides_defaults():
    raw = {"hidden_size": 32, "num_layers": 2, "epochs": 5}
    config = GRUConfig.from_dict(num_features=8, raw=raw)
    assert config.num_features == 8
    assert config.hidden_size == 32
    assert config.num_layers == 2
    assert config.epochs == 5
    assert config.lookback == 60  # untouched default


def test_config_from_dict_ignores_unknown_keys():
    raw = {"hidden_size": 32, "unrelated_key": "ignored"}
    config = GRUConfig.from_dict(num_features=5, raw=raw)
    assert config.hidden_size == 32
    assert not hasattr(config, "unrelated_key")


# ---------------------------------------------------------------------
# GRUModel - shape and forward-pass correctness
# ---------------------------------------------------------------------

@pytest.fixture
def small_config():
    return GRUConfig(num_features=6, lookback=10, hidden_size=8, num_layers=1, dropout=0.2)


@pytest.fixture
def small_stacked_config():
    """num_layers > 1 exercises nn.GRU's internal inter-layer dropout path."""
    return GRUConfig(num_features=6, lookback=10, hidden_size=8, num_layers=2, dropout=0.3)


def test_forward_output_shape(small_config):
    model = GRUModel(small_config)
    x = torch.randn(4, small_config.lookback, small_config.num_features)
    out = model(x)
    assert out.shape == (4, 1)


def test_forward_output_shape_stacked(small_stacked_config):
    model = GRUModel(small_stacked_config)
    x = torch.randn(4, small_stacked_config.lookback, small_stacked_config.num_features)
    out = model(x)
    assert out.shape == (4, 1)


def test_forward_rejects_wrong_ndim(small_config):
    model = GRUModel(small_config)
    x_2d = torch.randn(4, small_config.num_features)
    with pytest.raises(ValueError):
        model(x_2d)


def test_forward_rejects_wrong_feature_count(small_config):
    model = GRUModel(small_config)
    x = torch.randn(4, small_config.lookback, small_config.num_features + 1)
    with pytest.raises(ValueError):
        model(x)


def test_gradients_flow_through_all_parameters(small_config):
    model = GRUModel(small_config)
    x = torch.randn(4, small_config.lookback, small_config.num_features)
    y = torch.randn(4, 1)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.any(param.grad != 0), f"{name} gradient is all zero"


def test_dropout_disabled_in_eval_mode(small_config):
    model = GRUModel(small_config)
    model.eval()
    x = torch.randn(2, small_config.lookback, small_config.num_features)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2)


def test_dropout_active_in_train_mode_changes_output():
    config = GRUConfig(num_features=6, lookback=10, hidden_size=32, num_layers=1, dropout=0.9)
    model = GRUModel(config)
    model.train()
    x = torch.randn(2, config.lookback, config.num_features)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1, out2)


def test_single_batch_item_works(small_config):
    model = GRUModel(small_config)
    x = torch.randn(1, small_config.lookback, small_config.num_features)
    out = model(x)
    assert out.shape == (1, 1)


# ---------------------------------------------------------------------
# GRU vs LSTM parity — same interface, comparable parameter footprint
# ---------------------------------------------------------------------

def test_gru_has_fewer_parameters_than_equivalent_lstm(small_config):
    """
    Sanity check on the 'GRU is lighter than LSTM' claim from the spec:
    with the SAME hidden_size/num_layers, a GRU cell has fewer weights
    than an LSTM cell (3 gates vs 4), so total parameter count must be
    strictly lower for an apples-to-apples comparison.
    """
    from src.models.lstm import LSTMConfig, LSTMModel

    lstm_config = LSTMConfig(
        num_features=small_config.num_features,
        lookback=small_config.lookback,
        hidden_size=small_config.hidden_size,
        num_layers=2,  # LSTMModel always stacks 2 by architecture
        dropout=small_config.dropout,
    )
    gru_model = GRUModel(small_config)
    lstm_model = LSTMModel(lstm_config)

    gru_params = sum(p.numel() for p in gru_model.parameters())
    lstm_params = sum(p.numel() for p in lstm_model.parameters())

    assert gru_params < lstm_params