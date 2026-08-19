# tests/test_train_batch.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_batch import BatchResult, run_batch, run_single_training


def _fake_runner(fail_on: set[tuple[str, str]]):
    """Builds a fake runner that fails only for specific (symbol, model)
    combinations -- lets tests control exactly which combos succeed
    without spawning any real subprocess or needing a real train.py."""

    def runner(symbol: str, model: str) -> BatchResult:
        success = (symbol, model) not in fail_on
        return BatchResult(
            symbol=symbol,
            model=model,
            success=success,
            exit_code=0 if success else 1,
            duration_seconds=0.01,
            stderr_tail="" if success else "no data for symbol",
        )

    return runner


def test_run_batch_all_succeed():
    runner = _fake_runner(fail_on=set())
    results = run_batch(["INTC", "AAPL"], ["lstm"], runner=runner)

    assert len(results) == 2
    assert all(r.success for r in results)


def test_run_batch_covers_every_symbol_model_combination():
    runner = _fake_runner(fail_on=set())
    results = run_batch(["INTC", "AAPL"], ["lstm", "gru"], runner=runner)

    combos = {(r.symbol, r.model) for r in results}
    assert combos == {
        ("INTC", "lstm"), ("INTC", "gru"),
        ("AAPL", "lstm"), ("AAPL", "gru"),
    }


def test_run_batch_continues_after_one_failure():
    """The core requirement: one bad symbol must NOT stop the rest of
    the batch from running."""
    runner = _fake_runner(fail_on={("BADSYMBOL", "lstm")})
    results = run_batch(["INTC", "BADSYMBOL", "AAPL"], ["lstm"], runner=runner)

    assert len(results) == 3  # all three were attempted, none skipped
    statuses = {r.symbol: r.success for r in results}
    assert statuses == {"INTC": True, "BADSYMBOL": False, "AAPL": True}


def test_run_batch_reports_failure_reason():
    runner = _fake_runner(fail_on={("BADSYMBOL", "lstm")})
    results = run_batch(["BADSYMBOL"], ["lstm"], runner=runner)

    failed = results[0]
    assert failed.success is False
    assert failed.stderr_tail == "no data for symbol"


def test_run_batch_empty_symbols_returns_empty_results():
    runner = _fake_runner(fail_on=set())
    results = run_batch([], ["lstm"], runner=runner)
    assert results == []


# ---------------------------------------------------------------------
# run_single_training - real subprocess plumbing (smoke test only,
# using `python -c` as a stand-in "train.py" so this doesn't depend on
# your actual training pipeline or database).
# ---------------------------------------------------------------------

def test_run_single_training_reports_success_for_exit_0(tmp_path):
    fake_train_script = tmp_path / "fake_train.py"
    fake_train_script.write_text(
        "import sys; print('ok'); sys.exit(0)"
    )
    result = run_single_training(
        symbol="INTC", model="lstm", train_script=str(fake_train_script)
    )
    assert result.success is True
    assert result.exit_code == 0


def test_run_single_training_reports_failure_for_nonzero_exit(tmp_path):
    fake_train_script = tmp_path / "fake_train.py"
    fake_train_script.write_text(
        "import sys; print('boom', file=sys.stderr); sys.exit(1)"
    )
    result = run_single_training(
        symbol="INTC", model="lstm", train_script=str(fake_train_script)
    )
    assert result.success is False
    assert result.exit_code == 1
    assert "boom" in result.stderr_tail
