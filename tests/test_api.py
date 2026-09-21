# tests/test_api.py

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, app_state
from api.routes.prediction import get_inference_service
from api.routes.market import get_repository
from src.storage.database import create_engine, init_db
from src.storage.repository import MarketDataRepository


class FakePredictor:
    metadata = {
        "symbol": "INTC",
        "model_name": "lstm",
        "validation_metrics": {"mae": 0.1},
        "test_metrics": {"mae": 0.2},
    }


class FakeInferenceService:
    def get_predictor(self, symbol, model_name="lstm"):
        if symbol != "INTC":
            raise FileNotFoundError(f"No model found for {symbol}")
        return FakePredictor()


def _repository(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'api-test.db'}")
    init_db(engine)
    repository = MarketDataRepository(engine)
    repository.upsert_batch(
        pd.DataFrame(
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
    )
    return repository


def test_new_api_routes_return_valid_data_and_404_for_unknown_symbols(tmp_path):
    repository = _repository(tmp_path)
    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_inference_service] = lambda: FakeInferenceService()

    try:
        with TestClient(app) as client:
            market = client.get("/api/market/INTC")
            model = client.get("/api/model/INTC")
            metrics = client.get("/api/metrics/INTC")

            assert market.status_code == 200
            assert market.json()["symbol"] == "INTC"
            assert model.status_code == 200
            assert model.json()["model_name"] == "lstm"
            assert metrics.status_code == 200
            assert metrics.json()["test_metrics"]["mae"] == 0.2

            assert client.get("/api/market/UNKNOWN").status_code == 404
            assert client.get("/api/model/UNKNOWN").status_code == 404
            assert client.get("/api/metrics/UNKNOWN").status_code == 404
    finally:
        app.dependency_overrides.clear()
        app_state.repository = None
        app_state.inference_service = None
