# tests/test_lstm.py
import pytest
import torch

from src.models.lstm import LSTMConfig, LSTMModel


# ---------------------------------------------------------------------
# LSTMConfig
# ---------------------------------------------------------------------

def test_config_defaults_match_spec():
    config = LSTMConfig(num_features=10)
    assert config.lookback == 60
    assert config.hidden_size == 128
    assert config.num_layers == 2
    assert config.dropout == 0.2
    assert config.learning_rate == 0.001
    assert config.batch_size == 64
    assert config.epochs == 50


def test_config_from_dict_overrides_defaults():
    raw = {"hidden_size": 64, "dropout": 0.3, "epochs": 10}
    config = LSTMConfig.from_dict(num_features=12, raw=raw)
    assert config.num_features == 12
    assert config.hidden_size == 64
    assert config.dropout == 0.3
    assert config.epochs == 10
    # Unspecified fields keep their defaults -- not hard-coded overrides.
    assert config.lookback == 60


def test_config_from_dict_ignores_unknown_keys():
    raw = {"hidden_size": 64, "some_unrelated_key": "ignored"}
    config = LSTMConfig.from_dict(num_features=5, raw=raw)
    assert config.hidden_size == 64
    assert not hasattr(config, "some_unrelated_key")


# ---------------------------------------------------------------------
# LSTMModel - shape and forward-pass correctness
# ---------------------------------------------------------------------

@pytest.fixture
def small_config():
    # Small dims on purpose -- this test suite verifies architecture
    # correctness, not training quality, so keep it fast.
    return LSTMConfig(
        num_features=6,
        lookback=10,
        hidden_size=8,
        num_layers=2,
        dropout=0.2,
    )


def test_forward_output_shape(small_config):
    model = LSTMModel(small_config)
    batch_size = 4
    x = torch.randn(batch_size, small_config.lookback, small_config.num_features)
    out = model(x)
    assert out.shape == (batch_size, 1)


def test_forward_rejects_wrong_ndim(small_config):
    model = LSTMModel(small_config)
    x_2d = torch.randn(4, small_config.num_features)  # missing lookback dim
    with pytest.raises(ValueError):
        model(x_2d)


def test_forward_rejects_wrong_feature_count(small_config):
    model = LSTMModel(small_config)
    wrong_features = small_config.num_features + 1
    x = torch.randn(4, small_config.lookback, wrong_features)
    with pytest.raises(ValueError):
        model(x)


def test_gradients_flow_through_all_parameters(small_config):
    """Verifies the architecture is actually trainable: a backward pass
    must produce a non-None, non-zero gradient for every parameter.
    Catches silent architecture bugs (e.g. a layer that's built but
    never used in forward())."""
    model = LSTMModel(small_config)
    x = torch.randn(4, small_config.lookback, small_config.num_features)
    y = torch.randn(4, 1)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.any(param.grad != 0), f"{name} gradient is all zero"


def test_dropout_disabled_in_eval_mode(small_config):
    """With dropout=0 disabled at eval(), the same input must produce
    the IDENTICAL output on repeated calls (no stochasticity). This
    would fail if eval() were not properly propagated to the dropout
    layer -- a common bug when composing custom modules."""
    model = LSTMModel(small_config)
    model.eval()
    x = torch.randn(2, small_config.lookback, small_config.num_features)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2)


def test_dropout_active_in_train_mode_changes_output(small_config):
    """Sanity check that dropout is actually wired in: in train() mode
    with dropout > 0, repeated forward passes on the same input should
    generally differ (stochastic). Uses a large hidden_size/dropout in
    this fixture to make the test reliably non-flaky."""
    config = LSTMConfig(
        num_features=6, lookback=10, hidden_size=32, num_layers=2, dropout=0.9
    )
    model = LSTMModel(config)
    model.train()
    x = torch.randn(2, config.lookback, config.num_features)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1, out2)


def test_single_batch_item_works(small_config):
    """batch_size=1 is an edge case some LSTM implementations mishandle."""
    model = LSTMModel(small_config)
    x = torch.randn(1, small_config.lookback, small_config.num_features)
    out = model(x)
    assert out.shape == (1, 1)