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

        raw_df = self._fetch_provider_data(symbol, fetch_start, fetch_end, interval)
        raw_df = self._ensure_symbol_column(raw_df, symbol)

        if raw_df.empty:
            return {
                "symbol": symbol,
                "fetched_rows": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "upserted_rows": 0,
                "last_timestamp": last_timestamp,
            }

        valid_df, invalid_df = self.validator.validate(raw_df)
        upserted_rows = self.repository.upsert_batch(valid_df) if not valid_df.empty else 0

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
    def _format_start(last_timestamp: Optional[datetime]) -> Optional[str]:
        if last_timestamp is None:
            return None
        return pd.Timestamp(last_timestamp).strftime("%Y-%m-%d")
