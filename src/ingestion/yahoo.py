## src/ingestion/yahoo.py

"""
Module: src/ingestion/yahoo.py
Description: Concrete implementation of `MarketDataSource` using Yahoo Finance API.
How it works:
    Communicates with external Yahoo Finance endpoints (via `yfinance` library or direct REST calls),
    retrieves raw market payload, handles error states/timeouts, parses raw columns, converts timestamps to UTC,
    and formats output into the standardized Data Contract schema.
"""

import pandas as pd
import yfinance as yf
from src.ingestion.base import MarketDataSource

CONTRACT_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


class YahooFinanceSource(MarketDataSource):
    """
    Market data provider implementation for Yahoo Finance.
    """

    def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")

        try:
            provider_end = pd.Timestamp(end).strftime("%Y-%m-%d")
            data = yf.download(
                tickers=symbol,
                start=start,
                end=provider_end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Yahoo Finance request failed for symbol '{symbol}': {exc}"
            ) from exc

        return self._normalize_data(data, symbol)

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")

        try:
            data = yf.Ticker(symbol).history(
                period="5d", interval="1d", auto_adjust=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"Yahoo Finance request failed for symbol '{symbol}': {exc}"
            ) from exc

        normalized = self._normalize_data(data, symbol)
        return normalized.tail(1).reset_index(drop=True)

    @staticmethod
    def _normalize_data(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError(
                f"Yahoo Finance returned no market data for symbol '{symbol}'"
            )

        frame = data.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            columns = {}
            for field in ("Open", "High", "Low", "Close", "Volume"):
                for column in frame.columns:
                    if field.lower() in {str(value).lower() for value in column}:
                        columns[field] = column
                        break
            missing = [
                field
                for field in ("Open", "High", "Low", "Close", "Volume")
                if field not in columns
            ]
            if missing:
                raise ValueError(
                    f"Yahoo Finance response for symbol '{symbol}' is missing columns: {missing}"
                )
            frame = frame[
                [columns[field] for field in ("Open", "High", "Low", "Close", "Volume")]
            ]
            frame.columns = ["open", "high", "low", "close", "volume"]
        else:
            frame = frame.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

        frame = frame.reset_index()
        timestamp_column = next(
            (
                column
                for column in ("timestamp", "Date", "Datetime")
                if column in frame.columns
            ),
            None,
        )
        if timestamp_column is None:
            raise ValueError(
                f"Yahoo Finance response for symbol '{symbol}' has no timestamp column"
            )

        required_price_columns = ["open", "high", "low", "close", "volume"]
        missing = [
            column for column in required_price_columns if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Yahoo Finance response for symbol '{symbol}' is missing columns: {missing}"
            )

        normalized = frame.rename(columns={timestamp_column: "timestamp"})
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
        for column in required_price_columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.dropna(subset=["timestamp", *required_price_columns])
        if normalized.empty:
            raise ValueError(
                f"Yahoo Finance returned no valid market data for symbol '{symbol}'"
            )

        normalized["symbol"] = symbol
        for column in ("open", "high", "low", "close"):
            normalized[column] = normalized[column].astype(float)
        normalized["volume"] = normalized["volume"].astype("int64")
        return (
            normalized[CONTRACT_COLUMNS].sort_values("timestamp").reset_index(drop=True)
        )
