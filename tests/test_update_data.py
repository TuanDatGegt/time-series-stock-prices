# tests/test_update_data.py

from unittest.mock import Mock

import pandas as pd

from scripts.update_data import UpdateDataPipeline
from src.ingestion.scheduler import IngestionScheduler


def test_scheduler_runs_update_pipeline_steps_in_order():
    calls = []
    frame = pd.DataFrame(
        [
            {
                "symbol": "INTC",
                "timestamp": "2026-08-01 09:30:00",
                "open": 25.0,
                "high": 26.0,
                "low": 24.5,
                "close": 25.8,
                "volume": 1000,
            }
        ]
    )
    ingestion = Mock()
    ingestion.sync_symbol.side_effect = lambda symbol: calls.append("ingest") or {}
    repository = Mock()
    repository.fetch_history.side_effect = lambda symbol: frame
    repository.upsert_batch.side_effect = lambda valid: calls.append("store") or 1
    validator = Mock()
    validator.validate.side_effect = lambda data: calls.append("validate") or (
        frame,
        frame.iloc[0:0],
    )
    feature_builder = Mock(
        side_effect=lambda data: calls.append("update_features") or data
    )
    predictor = Mock()
    predictor.predict_next.side_effect = lambda data: calls.append(
        "refresh_prediction"
    ) or {"predicted_price": 26.0}
    pipeline = UpdateDataPipeline(
        ingestion_service=ingestion,
        repository=repository,
        validator=validator,
        feature_builder=feature_builder,
        predictor_factory=lambda symbol: predictor,
        prediction_sink=Mock(),
    )
    scheduler = IngestionScheduler(pipeline, ["INTC"])

    scheduler._sync_job()

    assert calls == [
        "ingest",
        "validate",
        "store",
        "update_features",
        "refresh_prediction",
    ]
