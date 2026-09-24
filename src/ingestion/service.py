## src/ingestion/service.py

"""
Module: src/ingestion/service.py
Description: Incremental Market Data Ingestion Service.
How it works:
    This service connects a market data provider (e.g., Yahoo Finance) with the database repository
    and data validation pipeline. It queries the latest recorded timestamp (`last_timestamp`) for a symbol
    from the repository, fetches missing data ranges, filters out duplicates, validates records, and
    executes UPSERT batch operations to ensure data freshness without redundant downloads.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
import pandas as pd

from src.storage.repository import MarketDataRepository
from src.validation.market_data import MarketDataValidator


class IncrementalIngestionService:
    """
    Service responsible for incrementally syncing market data to prevent duplicate fetching.
    """

    def __init__(
        self,
        provider,
        repository: MarketDataRepository,
        validator: Optional[MarketDataValidator] = None,
    ):
        """
        Initialize the ingestion service with data provider, repository, and validator instances.
        """
        self.provider = provider
        self.repository = repository
        self.validator = validator or MarketDataValidator()

    def sync_symbol(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> dict:
        """
        Synchronize market data for a given symbol incrementally.

        Args:
            symbol (str): Target stock or asset ticker symbol.
            start (Optional[str]): Explicit start boundary date/timestamp.
            end (Optional[str]): Explicit end boundary date/timestamp.
            interval (str): Candle sampling frequency (e.g., '1d', '1h').

        Returns:
            dict: Execution metrics containing fetched_rows, valid_rows, invalid_rows,
                  upserted_rows, and last_timestamp.
        """
        # Fetch the latest recorded timestamp from the database repository
        last_timestamp = self.repository.get_last_timestamp(symbol)

        # Determine date boundaries based on last_timestamp or explicit inputs
        fetch_start = start or self._format_start(last_timestamp)
        fetch_end = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        fetch_start = self._normalize_start_bound(fetch_start)
        fetch_end = self._normalize_end_bound(fetch_end)

        print(f"[debug] last_timestamp={last_timestamp}")
        print(f"[debug] fetch_start={fetch_start}, fetch_end={fetch_end}")

        # Fetch raw data from provider API and format symbol/timestamp columns
        raw_df = self._fetch_provider_data(symbol, fetch_start, fetch_end, interval)
        raw_df = self._ensure_symbol_column(raw_df, symbol)
        if "timestamp" in raw_df.columns:
            raw_df["timestamp"] = pd.to_datetime(
                raw_df["timestamp"], errors="coerce", utc=True
            )
        print(f"[debug] raw_df rows before validation:\n{raw_df}")

        # Return early if no records were returned by the provider
        if raw_df.empty:
            return {
                "symbol": symbol,
                "fetched_rows": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "upserted_rows": 0,
                "last_timestamp": last_timestamp,
            }

        # Filter dataset by explicit start parameter if provided
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            raw_df = raw_df[raw_df["timestamp"] >= start_ts].copy()

        # Filter dataset by explicit end parameter if provided
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            if end_ts == end_ts.normalize():
                end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            raw_df = raw_df[raw_df["timestamp"] <= end_ts].copy()

        if raw_df.empty:
            return {
                "symbol": symbol,
                "fetched_rows": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "upserted_rows": 0,
                "last_timestamp": last_timestamp,
            }

        # Exclude records already existing in the database (timestamp &lt;= last_timestamp)
        if last_timestamp is not None:
            last_ts = pd.Timestamp(last_timestamp)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            raw_df = raw_df[raw_df["timestamp"] > last_ts].copy()

        if raw_df.empty:
            return {
                "symbol": symbol,
                "fetched_rows": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "upserted_rows": 0,
                "last_timestamp": self.repository.get_last_timestamp(symbol),
            }

        # Run records through validation layer (OHLC check, null check, schema check)
        valid_df, invalid_df = self.validator.validate(raw_df)
        print(f"[debug] valid_df:\n{valid_df}")
        print(f"[debug] invalid_df:\n{invalid_df}")

        # Perform UPSERT into database repository for valid records
        upserted_rows = (
            self.repository.upsert_batch(valid_df) if not valid_df.empty else 0
        )
        print(f"[debug] upserted_rows={upserted_rows}")

        return {
            "symbol": symbol,
            "fetched_rows": len(raw_df),
            "valid_rows": len(valid_df),
            "invalid_rows": len(invalid_df),
            "upserted_rows": upserted_rows,
            "last_timestamp": self.repository.get_last_timestamp(symbol),
        }

    def _fetch_provider_data(
        self, symbol: str, start: Optional[str], end: str, interval: str
    ) -> pd.DataFrame:
        """
        Dynamically invoke provider data retrieval method (`fetch_historical` or `download`).
        """
        if hasattr(self.provider, "fetch_historical"):
            return self.provider.fetch_historical(
                symbol=symbol, start=start, end=end, interval=interval
            )
        if hasattr(self.provider, "download"):
            return self.provider.download(
                symbol=symbol, start=start, end=end, interval=interval
            )
        raise TypeError("Provider must expose fetch_historical() or download().")

    @staticmethod
    def _ensure_symbol_column(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Ensure the output DataFrame contains a valid 'symbol' column.
        """
        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

        normalized = df.copy()
        if "symbol" not in normalized.columns:
            normalized["symbol"] = symbol
        return normalized

    @staticmethod
    def _normalize_start_bound(value: Optional[str]) -> Optional[str]:
        """
        Format start boundary timestamp string to YYYY-MM-DD for yfinance compatibility.
        """
        if value is None:
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    @staticmethod
    def _normalize_end_bound(value: Optional[str]) -> Optional[str]:
        """
        Format end boundary timestamp string to cover the full target day.
        """
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp == timestamp.normalize():
            timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_start(last_timestamp: Optional[datetime]) -> Optional[str]:
        """
        Convert datetime object to formatted string YYYY-MM-DD.
        """
        if last_timestamp is None:
            return None
        return pd.Timestamp(last_timestamp).strftime("%Y-%m-%d")
