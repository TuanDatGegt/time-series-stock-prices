# src/training/early_stopping.py
"""
Phase 14 - Early stopping.

Deliberately separated from the training loop (src/training/train.py)
so the stop/save decision logic can be unit-tested with a scripted
sequence of validation losses, without needing a real model or real
data -- this determinism matters because early stopping bugs (off-by-one
on patience, saving the wrong epoch's checkpoint) are easy to introduce
and hard to catch by eye in a full training run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingResult:
    improved: bool          # did THIS step's val_loss beat the best seen so far?
    should_stop: bool       # has patience been exhausted?
    best_loss: float
    best_epoch: int
    epochs_without_improvement: int


class EarlyStopping:
    """
    Tracks validation loss across epochs. Call `.step(val_loss, epoch)`
    once per epoch. When `epochs_without_improvement > patience`,
    `should_stop` becomes True and training should halt -- the caller
    is responsible for having already saved the checkpoint on the last
    epoch where `improved` was True (this class does not touch disk).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        if patience < 0:
            raise ValueError("patience must be >= 0")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.best_epoch: int = -1
        self.epochs_without_improvement: int = 0

    def step(self, val_loss: float, epoch: int) -> EarlyStoppingResult:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = self.epochs_without_improvement > self.patience

        return EarlyStoppingResult(
            improved=improved,
            should_stop=should_stop,
            best_loss=self.best_loss,
            best_epoch=self.best_epoch,
            epochs_without_improvement=self.epochs_without_improvement,
        )