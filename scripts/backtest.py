## scripts/backtest.py

"""
Script: scripts/backtest.py
Description: CLI Execution Script for Walk-Forward Model Backtesting.
How it works:
    Parses CLI arguments (--symbol, --model, --n-folds), loads historical market data from repository,
    triggers `WalkForwardBacktester`, prints backtest progress per fold, and displays aggregated average scores.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.builder import build_features
from src.models.baseline import NaiveBaseline
from src.storage.database import create_engine
from src.storage.repository import MarketDataRepository
from src.training.backtest import WalkForwardBacktester
from src.utils.config import load_config
from src.utils.logger import get_logger


def baseline_trainer_wrapper(
    train_df, test_df, lookback, horizon, feature_cols, target_col
):
    """
    Wrapper function to train and predict NaiveBaseline model for backtesting fold.
    """
    model = NaiveBaseline(close_col=target_col)
    model.fit(train_df)
    target = test_df[target_col].shift(-horizon)
    valid_mask = target.notna().to_numpy()
    if not valid_mask.any():
        raise ValueError("Test fold has no target rows for the requested horizon")

    predictions = model.predict(test_df).astype(float)[valid_mask]
    y_true = target.to_numpy()[valid_mask]
    prev_close = test_df[target_col].to_numpy()[valid_mask]

    return "naive", y_true, predictions, prev_close


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest on market data"
    )
    parser.add_argument(
        "--symbol", required=True, help="Market ticker symbol, e.g. INTC"
    )
    parser.add_argument(
        "--n-folds", type=int, default=3, help="Number of expanding backtest folds"
    )
    parser.add_argument(
        "--config-dir", default="configs", help="Directory containing config files"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = get_logger()

    try:
        config = load_config(Path(args.config_dir))
        database_url = config.get("database_url")
        if not database_url:
            raise ValueError("database_url is required in config.yaml")

        repository = MarketDataRepository(create_engine(database_url))
        raw_df = repository.fetch_history(args.symbol.upper())

        if raw_df.empty:
            raise ValueError(f"No market data found for symbol {args.symbol}")

        # Build technical indicator features
        featured_df = build_features(raw_df)
        feature_cols = config["training"]["feature_columns"]
        target_col = config["training"]["target_column"]

        logger.info(
            "Starting Walk-Forward Backtest for %s across %d folds...",
            args.symbol,
            args.n_folds,
        )

        backtester = WalkForwardBacktester(
            lookback=config["training"]["lookback"],
            horizon=config["training"]["horizon"],
            n_folds=args.n_folds,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        summary_df, _ = backtester.run_backtest(featured_df, baseline_trainer_wrapper)

        logger.info("\n--- BACKTEST RESULTS SUMMARY ---")
        logger.info("\n%s", summary_df.to_string(index=False))
        logger.info(
            "Average MAE: %.4f | Average RMSE: %.4f",
            summary_df["mae"].mean(),
            summary_df["rmse"].mean(),
        )

        return 0

    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
