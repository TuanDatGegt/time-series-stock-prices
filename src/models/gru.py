## src/models/gru.py
"""
Phase 12 - GRU model.

Per spec: GRU is a lighter benchmark alongside LSTM (Naive -> XGBoost ->
LSTM -> GRU comparison chain from Phase 10/11). The spec diagram for
this phase is intentionally simpler than LSTM's ("GRU" alone, vs LSTM's
"Input -> LSTM -> Dropout -> LSTM -> Dense"), reflecting GRU's typical
role as a lighter-weight alternative -- so this defaults to a SINGLE
GRU layer (num_layers=1) rather than LSTM's stacked-2-layer default.
num_layers stays configurable via GRUConfig in case you want to test a
stacked GRU too, for a fair side-by-side against LSTM.

Consumes the same sequences as LSTM (src/preprocessing/sequence.py):
    X.shape = (samples, lookback, num_features)
    y.shape = (samples, 1)

Hyperparameters are NOT hard-coded -- see GRUConfig, loaded from
configs/model.yaml the same way as LSTMConfig.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GRUConfig:
    """
    Mirrors LSTMConfig's shape, with a lighter default (num_layers=1,
    smaller hidden_size) matching GRU's role as the lightweight
    benchmark. Expected configs/model.yaml block:

        gru:
          lookback: 60
          hidden_size: 64
          num_layers: 1
          dropout: 0.2
          learning_rate: 0.001
          batch_size: 64
          epochs: 50
    """

    num_features: int  # supplied at runtime, not from model.yaml (depends
    # on Phase 7's feature set, same as LSTMConfig)
    lookback: int = 60
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50

    @classmethod
    def from_dict(cls, num_features: int, raw: dict) -> "GRUConfig":
        """Build from the dict loaded via src.utils.config (model.yaml)."""
        known_fields = {f for f in cls.__dataclass_fields__ if f != "num_features"}
        filtered = {k: v for k, v in raw.items() if k in known_fields}
        return cls(num_features=num_features, **filtered)


class GRUModel(nn.Module):
    """
    Input -> GRU -> Dropout -> Dense -> Prediction

    Uses nn.GRU's built-in num_layers/dropout support directly (unlike
    LSTMModel, which manually stacks two separate nn.LSTM instances to
    match the spec's explicit "LSTM -> Dropout -> LSTM" diagram). For
    GRU the spec doesn't call for a second stacked layer by default, so
    a single nn.GRU(num_layers=config.num_layers) is used, with an
    external Dropout applied to its output before the final Dense --
    if num_layers > 1 is configured, nn.GRU's own inter-layer dropout
    is also enabled automatically for a genuinely deeper benchmark.
    """

    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            # nn.GRU only applies its internal dropout BETWEEN layers,
            # and only takes effect when num_layers > 1 (PyTorch would
            # otherwise warn/ignore it for num_layers == 1).
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, num_features)
        returns: (batch, 1) -- next-day close prediction
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected input of shape (batch, lookback, num_features), "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.config.num_features:
            raise ValueError(
                f"Expected {self.config.num_features} features in the last "
                f"dimension, got {x.shape[-1]}"
            )

        out, h_n = self.gru(x)  # out: (batch, lookback, hidden_size)
        out = self.dropout(out)

        last_step = out[:, -1, :]  # (batch, hidden_size)
        prediction = self.dense(last_step)  # (batch, 1)
        return prediction
