# src/models/lstm.py
"""
Phase 11 - LSTM model.

Architecture (per spec):
    Input -> LSTM -> Dropout -> LSTM -> Dense -> Prediction

Consumes sequences produced by src/preprocessing/sequence.py:
    X.shape = (samples, lookback, num_features)
    y.shape = (samples, 1)

All hyperparameters (lookback, hidden_size, num_layers, dropout,
learning_rate, batch_size, epochs) are NOT hard-coded here -- they are
passed in via LSTMConfig, which is constructed from configs/model.yaml
by src/utils/config.py. This module only defines architecture; training
loop (optimizer, epochs, early stopping) belongs to Phase 13/14.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    """
    Mirrors the hyperparameter block in configs/model.yaml:

        lookback: 60
        hidden_size: 128
        num_layers: 2
        dropout: 0.2
        learning_rate: 0.001
        batch_size: 64
        epochs: 50

    learning_rate / batch_size / epochs are consumed by the training
    pipeline (Phase 13), not by this module directly -- they are kept
    here anyway so the whole hyperparameter set has ONE typed home and
    nothing has to be duplicated/hard-coded in train.py either.
    """

    num_features: int  # must be supplied at runtime (depends on the
                        # feature set built in Phase 7); NOT in model.yaml
    lookback: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50

    @classmethod
    def from_dict(cls, num_features: int, raw: dict) -> "LSTMConfig":
        """Build from the dict loaded via src.utils.config (model.yaml)."""
        known_fields = {f for f in cls.__dataclass_fields__ if f != "num_features"}
        filtered = {k: v for k, v in raw.items() if k in known_fields}
        return cls(num_features=num_features, **filtered)


class LSTMModel(nn.Module):
    """
    Input -> LSTM -> Dropout -> LSTM -> Dense -> Prediction

    Two stacked LSTM layers (per spec diagram), dropout applied between
    them, final Dense layer maps the last timestep's hidden state to a
    single scalar prediction (next-day close, consistent with the
    target definition used in Phase 10's baselines).
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        # First LSTM layer: num_features -> hidden_size
        self.lstm1 = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=config.dropout)

        # Second LSTM layer: hidden_size -> hidden_size
        self.lstm2 = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )

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

        out, _ = self.lstm1(x)           # (batch, lookback, hidden_size)
        out = self.dropout(out)
        out, (h_n, _) = self.lstm2(out)  # (batch, lookback, hidden_size)

        # Use only the LAST timestep's output -- this is the standard
        # many-to-one setup matching y.shape = (samples, 1) from Phase 9.
        last_step = out[:, -1, :]        # (batch, hidden_size)
        prediction = self.dense(last_step)  # (batch, 1)
        return prediction

    