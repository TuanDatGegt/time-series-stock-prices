from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.storage.repository import MarketDataRepository
from src.validation.market_data import MarketDataValidator


class IncrementalIngestionService:
    def __init__(self, provider, repository: MarketDataRepository, validator: Optional[MarketDataValidator] = None):
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
        last_timestamp = self.repository.get_last_timestamp(symbol)
        fetch_start = start or self._format_start(last_timestamp)
        fetch_end = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        fetch_start = self._normalize_start_bound(fetch_start)
        fetch_end = self._normalize_end_bound(fetch_end)

        print(f"[debug] last_timestamp={last_timestamp}")
        print(f"[debug] fetch_start={fetch_start}, fetch_end={fetch_end}")

        raw_df = self._fetch_provider_data(symbol, fetch_start, fetch_end, interval)
        raw_df = self._ensure_symbol_column(raw_df, symbol)
        if "timestamp" in raw_df.columns:
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
        print(f"[debug] raw_df rows before validation:\n{raw_df}")

        if raw_df.empty:
            return {
                "symbol": symbol,
                "fetched_rows": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "upserted_rows": 0,
                "last_timestamp": last_timestamp,
            }

        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts == start_ts.normalize():
                start_ts = start_ts.normalize()
            raw_df = raw_df[raw_df["timestamp"] >= start_ts].copy()
            print(f"[debug] raw_df after filtering by start:\n{raw_df}")

        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts == end_ts.normalize():
                end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            raw_df = raw_df[raw_df["timestamp"] <= end_ts].copy()
            print(f"[debug] raw_df after filtering by end:\n{raw_df}")

        if raw_df.empty:
            return {
                "symbol": symbol,
                "fetched_rows": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "upserted_rows": 0,
                "last_timestamp": last_timestamp,
            }

        if last_timestamp is not None:
            raw_df = raw_df[raw_df["timestamp"] > pd.Timestamp(last_timestamp)].copy()
            print(f"[debug] raw_df after filtering by last_timestamp:\n{raw_df}")

        if raw_df.empty:
            return {
                "symbol": symbol,
                "fetched_rows": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "upserted_rows": 0,
                "last_timestamp": self.repository.get_last_timestamp(symbol),
            }

        valid_df, invalid_df = self.validator.validate(raw_df)
        print(f"[debug] valid_df:\n{valid_df}")
        print(f"[debug] invalid_df:\n{invalid_df}")

        upserted_rows = self.repository.upsert_batch(valid_df) if not valid_df.empty else 0
        print(f"[debug] upserted_rows={upserted_rows}")

        return {
            "symbol": symbol,
            "fetched_rows": len(raw_df),
            "valid_rows": len(valid_df),
            "invalid_rows": len(invalid_df),
            "upserted_rows": upserted_rows,
            "last_timestamp": self.repository.get_last_timestamp(symbol),
        }

    def _fetch_provider_data(self, symbol: str, start: Optional[str], end: str, interval: str) -> pd.DataFrame:
        if hasattr(self.provider, "fetch_historical"):
            return self.provider.fetch_historical(symbol=symbol, start=start, end=end, interval=interval)
        if hasattr(self.provider, "download"):
            return self.provider.download(symbol=symbol, start=start, end=end, interval=interval)
        raise TypeError("Provider must expose fetch_historical() or download().")

    @staticmethod
    def _ensure_symbol_column(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])

        normalized = df.copy()
        if "symbol" not in normalized.columns:
            normalized["symbol"] = symbol
        return normalized

    @staticmethod
    def _normalize_start_bound(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if ts == ts.normalize():
            return ts.strftime("%Y-%m-%d 00:00:00")
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _normalize_end_bound(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if ts == ts.normalize():
            return (ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_start(last_timestamp: Optional[datetime]) -> Optional[str]:
        if last_timestamp is None:
            return None
        return pd.Timestamp(last_timestamp).strftime("%Y-%m-%d")
