## scripts/run_scheduler.py
"""
Script: scripts/run_scheduler.py
Description: CLI Worker Entrypoint for Automated Market Data Scheduler.
How it works:
    Loads project configuration (configs/config.yaml), connects to the SQL database,
    instantiates YahooFinanceSource and IncrementalIngestionService, and keeps the
    IngestionScheduler running continuously.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.scheduler import IngestionScheduler
from scripts.update_data import build_pipeline
from src.utils.config import load_config
from src.utils.logger import get_logger


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for scheduler worker."""
    parser = argparse.ArgumentParser(
        description="Run background market data ingestion scheduler"
    )
    parser.add_argument(
        "--symbols",
        default="INTC,AAPL,MSFT",
        help="Comma-separated list of target stock tickers to monitor",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Sync interval frequency in minutes (default: 15)",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing configuration YAML files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution loop."""
    args = build_parser().parse_args(argv)
    logger = get_logger("scheduler_worker")

    try:
        config = load_config(Path(args.config_dir))
        db_url = config.get("database_url", "sqlite:///data/forecasting.db")

        pipeline = build_pipeline(config)

        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

        logger.info("Starting Scheduler Worker process for symbols: %s", symbols)
        scheduler = IngestionScheduler(
            ingestion_service=pipeline,
            symbols=symbols,
            interval_minutes=args.interval,
        )

        scheduler.start(run_immediately=True)

        # Keep main thread alive
        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler worker process interrupted. Exiting...")
        return 0
    except Exception as exc:
        logger.error("Scheduler worker failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
