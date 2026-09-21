# tests/test_yahoo.py

from src.ingestion.yahoo import YahooFinanceSource

CONTRACT_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_historical_returns_aapl_contract_from_yahoo():
    data = YahooFinanceSource().fetch_historical(
        symbol="AAPL",
        start="2025-01-02",
        end="2025-01-10",
        interval="1d",
    )

    assert list(data.columns) == CONTRACT_COLUMNS
    assert len(data) > 0
    assert set(data["symbol"]) == {"AAPL"}
    assert data["timestamp"].dt.tz is not None
    assert data[["open", "high", "low", "close", "volume"]].notna().all().all()


def test_fetch_latest_returns_one_aapl_record_from_yahoo():
    data = YahooFinanceSource().fetch_latest("AAPL")

    assert list(data.columns) == CONTRACT_COLUMNS
    assert len(data) == 1
    assert data.iloc[0]["symbol"] == "AAPL"
