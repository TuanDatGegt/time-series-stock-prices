from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_logger


SUPPORTED_MODELS = ("lstm", "gru", "xgboost")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train a time-series forecasting model")
	parser.add_argument("--symbol", required=True, help="Market symbol, for example INTC")
	parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
	parser.add_argument("--config-dir", default="configs", help="Directory containing YAML config files")
	parser.add_argument("--checkpoint-dir", default=None)
	parser.add_argument("--metadata-dir", default=None)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	logger = get_logger()
	try:
		from src.training.train import run_training
		from src.utils.config import load_config

		config = load_config(Path(args.config_dir))
		result = run_training(
			symbol=args.symbol,
			model_name=args.model,
			config=config,
			checkpoint_dir=args.checkpoint_dir,
			metadata_dir=args.metadata_dir,
			logger=logger,
		)
		logger.info("Validation metrics: %s", result.validation_metrics)
		logger.info("Test metrics: %s", result.test_metrics)
		logger.info("Registered metadata: %s", result.metadata_path)
		return 0
	except Exception as exc:
		logger.error("Training failed: %s", exc)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
