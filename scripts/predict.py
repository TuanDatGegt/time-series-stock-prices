## scripts/predict.py
"""
Script: scripts/predict.py
Description: CLI Command Script for Running Price Predictions.
How it works:
    Parses CLI arguments (--symbol, --model, --config-dir), connects to database repository,
    invokes `InferenceService` to predict next price movement, and logs formatted output to terminal.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.inference.service import InferenceService
from src.storage.database import create_engine
from src.storage.repository import MarketDataRepository
from src.utils.config import load_config
from src.utils.logger import get_logger


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for prediction script."""
    parser = argparse.ArgumentParser(
        description="Generate price forecast for a market symbol"
    )
    parser.add_argument(
        "--symbol", required=True, help="Stock ticker symbol (e.g. INTC)"
    )
    parser.add_argument(
        "--model", default="lstm", choices=["lstm", "gru", "xgboost"], help="Model name"
    )
    parser.add_argument(
        "--config-dir", default="configs", help="Directory containing config YAML files"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution flow."""
    args = build_parser().parse_args(argv)
    logger = get_logger()

    try:
        config = load_config(Path(args.config_dir))
        database_url = config.get("database_url")
        if not database_url:
            raise ValueError("database_url is required in config.yaml")

        # Initialize storage repository and inference service
        engine = create_engine(database_url)
        repository = MarketDataRepository(engine)

        checkpoint_dir = config.get("paths", {}).get(
            "checkpoint_dir", "models/checkpoints"
        )
        metadata_dir = config.get("paths", {}).get("metadata_dir", "models/metadata")

        service = InferenceService(
            repository=repository,
            checkpoint_dir=checkpoint_dir,
            metadata_dir=metadata_dir,
        )

        logger.info(
            "Executing price forecast for %s using [%s]...",
            args.symbol.upper(),
            args.model.lower(),
        )
        result = service.predict_symbol(symbol=args.symbol, model_name=args.model)

        logger.info("\n--- PREDICTION RESULT ---")
        logger.info("Symbol:               %s", result["symbol"])
        logger.info("Timestamp:            %s", result["timestamp"])
        logger.info("Current Price:        $%.4f", result["current_price"])
        logger.info("Predicted Price:      $%.4f", result["predicted_price"])
        logger.info("Expected Change:      %+.2f%%", result["predicted_change_pct"])
        logger.info("Predicted Direction:  %s", result["direction"])
        logger.info(
            "Model Info:           %s (v%s)",
            result["model_name"],
            result["model_version"],
        )

        return 0

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
