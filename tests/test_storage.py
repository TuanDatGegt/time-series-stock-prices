# tests/test_storage.py

import pandas as pd

from src.storage.database import create_engine, init_db
from src.storage.repository import MarketDataRepository

SAMPLE_DATA = pd.DataFrame(
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


def test_repository_upsert_and_history_query():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    repo = MarketDataRepository(engine)

    inserted = repo.upsert_batch(SAMPLE_DATA)

    assert inserted == 2
    assert repo.get_last_timestamp("INTC") is not None

    history = repo.fetch_history("INTC")
    assert len(history) == 2
    assert history["symbol"].tolist() == ["INTC", "INTC"]


def test_repository_updates_existing_rows_without_duplicates():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    repo = MarketDataRepository(engine)

    repo.upsert_batch(SAMPLE_DATA)
    update_df = SAMPLE_DATA.copy()
    update_df.loc[0, "close"] = 27.5

    rows_upserted = repo.upsert_batch(update_df)

    assert rows_upserted == 1
    latest = repo.fetch_history("INTC")
    assert latest["close"].max() == 27.5
