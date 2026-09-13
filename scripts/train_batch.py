# scripts/train_batch.py

"""
Batch training wrapper for Phase 13.

    python scripts/train_batch.py --symbols INTC,AAPL,MSFT --models lstm,gru

This does NOT reimplement or modify scripts/train.py -- it calls the
EXISTING single-symbol CLI once per (symbol, model) combination via
subprocess. This keeps the two scripts fully decoupled: train_batch.py
has zero knowledge of train.py's internals (its argparse flags are the
only contract), so it stays correct even if train.py's internal
implementation changes.

Design choices:
    - One symbol/model failing (e.g. missing DB data for that symbol)
      does NOT stop the batch -- every combination is attempted, and a
      summary table at the end shows which succeeded/failed. This
      matters because a single bad symbol shouldn't block training the
      other 19 you actually have data for.
    - Exit code is 0 only if ALL combinations succeeded, 1 otherwise --
      so this is safe to use in a CI/cron job (Phase 26-27 scheduler)
      without silently masking partial failures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class BatchResult:
    symbol: str
    model: str
    success: bool
    exit_code: int
    duration_seconds: float
    stderr_tail: str = ""  # last bit of stderr, for quick diagnosis in the summary


def run_single_training(
    symbol: str,
    model: str,
    python_executable: str = sys.executable,
    train_script: str = "scripts/train.py",
) -> BatchResult:
    """Invokes the existing single-symbol CLI as a subprocess. This is
    the ONLY function that actually shells out -- kept separate from
    run_batch() so tests can inject a fake runner instead of spawning
    real subprocesses."""
    start = time.monotonic()
    result = subprocess.run(
        [python_executable, train_script, "--symbol", symbol, "--model", model],
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - start

    stderr_tail = (
        result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
    )

    return BatchResult(
        symbol=symbol,
        model=model,
        success=(result.returncode == 0),
        exit_code=result.returncode,
        duration_seconds=duration,
        stderr_tail=stderr_tail,
    )


def run_batch(
    symbols: list[str],
    models: list[str],
    runner: Callable[[str, str], BatchResult] = run_single_training,
) -> list[BatchResult]:
    """
    Trains every (symbol, model) combination. Continues past individual
    failures -- does not raise, does not stop early. Returns the full
    list of results (both successes and failures) for the caller to
    summarize/report on.
    """
    results: list[BatchResult] = []
    for symbol in symbols:
        for model in models:
            result = runner(symbol, model)
            results.append(result)
    return results


def print_summary(results: list[BatchResult]) -> None:
    total = len(results)
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n{'=' * 60}")
    print(f"Batch training summary: {len(succeeded)}/{total} succeeded")
    print(f"{'=' * 60}")
    for r in results:
        status = "OK  " if r.success else "FAIL"
        line = f"[{status}] {r.symbol:<8} {r.model:<6} ({r.duration_seconds:.1f}s)"
        if not r.success and r.stderr_tail:
            line += f"  -- {r.stderr_tail}"
        print(line)

    if failed:
        print(f"\n{len(failed)} combination(s) failed:")
        for r in failed:
            print(f"  - {r.symbol} / {r.model} (exit code {r.exit_code})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiple symbol/model combinations by calling scripts/train.py."
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated ticker symbols, e.g. INTC,AAPL,MSFT",
    )
    parser.add_argument(
        "--models",
        default="lstm",
        help="Comma-separated model names, e.g. lstm,gru (default: lstm)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not symbols:
        print("Error: --symbols must contain at least one symbol", file=sys.stderr)
        sys.exit(1)

    print(
        f"Training {len(symbols)} symbol(s) x {len(models)} model(s) = "
        f"{len(symbols) * len(models)} run(s)..."
    )

    results = run_batch(symbols, models)
    print_summary(results)

    all_succeeded = all(r.success for r in results)
    sys.exit(0 if all_succeeded else 1)


if __name__ == "__main__":
    main()
