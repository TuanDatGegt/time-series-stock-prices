# tests/test_early_stopping.py
import pytest

from src.training.early_stopping import EarlyStopping


def test_improving_losses_never_trigger_stop():
    es = EarlyStopping(patience=3)
    losses = [1.0, 0.8, 0.6, 0.4, 0.2]
    for epoch, loss in enumerate(losses):
        result = es.step(loss, epoch)
        assert result.improved is True
        assert result.should_stop is False
    assert es.best_loss == 0.2
    assert es.best_epoch == 4


def test_stops_after_patience_epochs_without_improvement():
    es = EarlyStopping(patience=3)
    # epoch 0: improves (best=1.0)
    # epochs 1-3: no improvement (3 epochs, exactly == patience) -> not yet stop
    # epoch 4: 4th epoch without improvement (> patience) -> stop
    losses = [1.0, 1.0, 1.0, 1.0, 1.0]
    results = [es.step(loss, epoch) for epoch, loss in enumerate(losses)]

    assert results[0].improved is True
    for r in results[1:4]:
        assert r.improved is False
        assert r.should_stop is False
    assert results[4].should_stop is True
    assert es.best_epoch == 0
    assert es.best_loss == 1.0


def test_best_checkpoint_is_not_always_the_last_epoch():
    """Directly verifies the spec's warning: 'the last checkpoint is not
    necessarily the best one'. Loss improves then gets worse -- best_epoch
    must point to the improvement, not to the final epoch."""
    es = EarlyStopping(patience=10)
    losses = [0.9, 0.5, 0.3, 0.6, 0.7, 0.8]  # best is epoch 2 (loss=0.3)
    for epoch, loss in enumerate(losses):
        es.step(loss, epoch)

    assert es.best_epoch == 2
    assert es.best_loss == 0.3


def test_min_delta_requires_meaningful_improvement():
    """A tiny improvement smaller than min_delta should NOT count as
    'improved' -- guards against noise-level fluctuations resetting the
    patience counter forever."""
    es = EarlyStopping(patience=2, min_delta=0.05)
    r1 = es.step(1.0, 0)
    assert r1.improved is True

    r2 = es.step(0.99, 1)  # improvement of 0.01 < min_delta=0.05
    assert r2.improved is False
    assert r2.epochs_without_improvement == 1


def test_rejects_negative_patience():
    with pytest.raises(ValueError):
        EarlyStopping(patience=-1)


def test_rejects_negative_min_delta():
    with pytest.raises(ValueError):
        EarlyStopping(patience=5, min_delta=-0.1)


def test_patience_zero_stops_immediately_on_first_non_improvement():
    es = EarlyStopping(patience=0)
    r1 = es.step(1.0, 0)  # improves, no stop
    assert r1.should_stop is False

    r2 = es.step(1.0, 1)  # no improvement, patience=0 -> stop immediately
    assert r2.should_stop is True