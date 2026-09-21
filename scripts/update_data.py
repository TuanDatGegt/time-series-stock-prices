## scripts/update_data.py

"""
Script: scripts/update_data.py
Description: CLI entrypoint for incremental data update runs.
How it works:
    Triggers `IncrementalIngestionService` to sync only newly published market bars since the latest
    recorded timestamp in PostgreSQL, minimizing network requests and preventing redundant database writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.builder import build_features
from src.inference.service import InferenceService
from src.ingestion.service import IncrementalIngestionService
from src.ingestion.yahoo import YahooFinanceSource
from src.storage.database import create_engine, init_db
from src.storage.repository import MarketDataRepository
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.validation.market_data import MarketDataValidator


class UpdateDataPipeline:
    """Orchestrate ingestion, validation, storage, features and prediction."""

    def __init__(
        self,
        ingestion_service: IncrementalIngestionService,
        repository: MarketDataRepository,
        validator: MarketDataValidator | None = None,
        feature_builder: Callable[[pd.DataFrame], pd.DataFrame] = build_features,
        predictor_factory: Callable[[str], Any] | None = None,
        prediction_sink: Callable[[str, dict[str, Any]], Any] | None = None,
        logger=None,
    ):
        self.ingestion_service = ingestion_service
        self.repository = repository
        self.validator = validator or MarketDataValidator()
        self.feature_builder = feature_builder
        self.predictor_factory = predictor_factory
        self.prediction_sink = prediction_sink
        self.logger = logger or get_logger("update_data")

    def run_symbol(self, symbol: str) -> dict[str, Any]:
        """Run all pipeline stages for one symbol and return stage results."""
        normalized_symbol = symbol.upper().strip()
        result: dict[str, Any] = {"symbol": normalized_symbol, "errors": {}}

        try:
            result["ingest"] = self.ingestion_service.sync_symbol(
                symbol=normalized_symbol
            )
        except Exception as exc:
            return self._stage_error(result, "ingest", exc)

        try:
            history = self.repository.fetch_history(normalized_symbol)
            valid, invalid = self.validator.validate(history)
            result["validate"] = {
                "valid_rows": len(valid),
                "invalid_rows": len(invalid),
            }
        except Exception as exc:
            return self._stage_error(result, "validate", exc)

        try:
            result["store"] = self.repository.upsert_batch(valid)
        except Exception as exc:
            return self._stage_error(result, "store", exc)

        try:
            result["features"] = self.feature_builder(valid)
        except Exception as exc:
            return self._stage_error(result, "update_features", exc)

        try:
            if self.predictor_factory is not None:
                predictor = self.predictor_factory(normalized_symbol)
                prediction = predictor.predict_next(history)
                result["prediction"] = prediction
                if self.prediction_sink is not None:
                    self.prediction_sink(normalized_symbol, prediction)
        except Exception as exc:
            return self._stage_error(result, "refresh_prediction", exc)

        return result

    def _stage_error(
        self, result: dict[str, Any], stage: str, error: Exception
    ) -> dict[str, Any]:
        message = f"{type(error).__name__}: {error}"
        result["errors"][stage] = message
        self.logger.error(
            "Update pipeline failed [%s] at %s: %s",
            result["symbol"],
            stage,
            error,
            exc_info=True,
        )
        return result


def save_latest_prediction(
    prediction_dir: str | Path, symbol: str, prediction: dict[str, Any]
) -> Path:
    """Persist the latest prediction payload for the dashboard/service layer."""
    destination = Path(prediction_dir) / symbol.upper()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "latest.json"
    path.write_text(json.dumps(prediction, indent=2, default=str), encoding="utf-8")
    return path


def build_pipeline(config: dict[str, Any]) -> UpdateDataPipeline:
    database_url = config.get("database_url", "sqlite:///data/forecasting.db")
    engine = create_engine(database_url)
    init_db(engine)
    repository = MarketDataRepository(engine)
    ingestion_service = IncrementalIngestionService(
        provider=YahooFinanceSource(), repository=repository
    )
    inference_service = InferenceService(
        repository=repository,
        checkpoint_dir=config.get("paths", {}).get(
            "checkpoint_dir", "models/checkpoints"
        ),
        metadata_dir=config.get("paths", {}).get("metadata_dir", "models/metadata"),
    )
    prediction_dir = config.get("paths", {}).get("prediction_dir", "data/predictions")
    return UpdateDataPipeline(
        ingestion_service=ingestion_service,
        repository=repository,
        predictor_factory=lambda symbol: inference_service.get_predictor(symbol),
        prediction_sink=lambda symbol, prediction: save_latest_prediction(
            prediction_dir, symbol, prediction
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run market data update pipeline")
    parser.add_argument("--symbols", default="INTC,AAPL,MSFT")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args(argv)
    logger = get_logger("update_data")
    pipeline = build_pipeline(load_config(Path(args.config_dir)))
    for symbol in (value.strip() for value in args.symbols.split(",")):
        if not symbol:
            continue
        result = pipeline.run_symbol(symbol)
        if result["errors"]:
            logger.error("Update failed for %s: %s", symbol, result["errors"])
        else:
            logger.info("Update complete for %s", symbol)
    return 0


if __name__ == "__main__":
    main()
