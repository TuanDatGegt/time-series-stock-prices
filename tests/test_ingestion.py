import pandas as pd

from src.ingestion.service import IncrementalIngestionService
from src.storage.database import create_engine, init_db
from src.storage.repository import MarketDataRepository


class FakeProvider:
    def __init__(self, rows):
        self.rows = rows

    def download(self, symbol, start=None, end=None, interval="1d"):
        filtered = []
        for row in self.rows:
            ts = pd.to_datetime(row["timestamp"])
            if start is not None and pd.Timestamp(start) > ts:
                continue
            if end is not None and ts > pd.Timestamp(end):
                continue
            filtered.append(row)
        return pd.DataFrame(filtered)


def test_incremental_insert_new_data():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    repo = MarketDataRepository(engine)

    repo.upsert_batch(
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

    provider = FakeProvider(
        [
            {
                "symbol": "INTC",
                "timestamp": "2026-08-01 09:30:00",
                "open": 25.0,
                "high": 26.0,
                "low": 24.5,
                "close": 25.8,
                "volume": 1000,
            },
            {
                "symbol": "INTC",
                "timestamp": "2026-08-02 09:30:00",
                "open": 25.8,
                "high": 26.2,
                "low": 25.5,
                "close": 26.1,
                "volume": 1200,
            },
        ]
    )

    service = IncrementalIngestionService(provider=provider, repository=repo)
    result = service.sync_symbol("INTC", start="2026-08-01", end="2026-08-02")

    assert result["upserted_rows"] == 1
    assert result["valid_rows"] == 1
    assert len(repo.fetch_history("INTC")) == 2


def test_incremental_rejects_invalid_row():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    repo = MarketDataRepository(engine)

    provider = FakeProvider(
        [
            {
                "symbol": "INTC",
                "timestamp": "2026-08-01 09:30:00",
                "open": 25.0,
                "high": 20.0,
                "low": 24.5,
                "close": 30.0,
                "volume": 1000,
            }
        ]
    )

    service = IncrementalIngestionService(provider=provider, repository=repo)
    result = service.sync_symbol("INTC", start="2026-08-01", end="2026-08-01")

    assert result["upserted_rows"] == 0
    assert result["invalid_rows"] == 1
    assert len(repo.fetch_history("INTC")) == 0


def test_incremental_no_duplicate():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    repo = MarketDataRepository(engine)

    provider = FakeProvider(
        [
            {
                "symbol": "INTC",
                "timestamp": "2026-08-01 09:30:00",
                "open": 25.0,
                "high": 26.0,
                "low": 24.5,
                "close": 25.8,
                "volume": 1000,
            },
            {
                "symbol": "INTC",
                "timestamp": "2026-08-02 09:30:00",
                "open": 25.8,
                "high": 26.2,
                "low": 25.5,
                "close": 26.1,
                "volume": 1200,
            },
        ]
    )

    service = IncrementalIngestionService(provider=provider, repository=repo)

    first = service.sync_symbol("INTC", start="2026-08-01", end="2026-08-02")
    second = service.sync_symbol("INTC", start="2026-08-01", end="2026-08-02")

    assert first["upserted_rows"] == 2
    assert second["upserted_rows"] == 0
    assert len(repo.fetch_history("INTC")) == 2
