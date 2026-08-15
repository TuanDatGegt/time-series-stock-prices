#src/ingestion/yahoo.py

"""Yahoo Finance Ingestion Client.
Responsibility is strictly: fetch raw market data and normalize it into the project's canonical schema;
It does NOT know about the storage - that is the job of "src/storage". This keeps the provider 
swappable: a structure 'PolygonClient' or 'AlphaVantageClient' only needs to implement the same 'download()' contract.
"""

from __future__ import annotations
from typing import Optional

import pandas as pd
import yfinance as yf

CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

class YahooFinanceClient:
    """Thin wrapper around  yfinance that returns a normalized DataFrame."""

    def download(self, symbol: str, start: str, end: Optional[str]=None, interval: str="1d") -> pd.DataFrame:
        """Download OHLCV history for `symbol` and normalize it.

        Returns a DataFrame with columns exactly:
        [timestamp, open, high, low, close, volume]
        sorted ascending by timestamp, with no index gaps in the
        underlying data structure (missing trading days are a
        validation concern, not an ingestion concern).
        """

        raw = yf.Ticket(symbol).history(
            start = start,
            end = end,
            interval = interval,
            auto_adjust = False
        )

        if raw is None or raw.empty:
            raise ValueError(
                f"no data returned for symbol={symbol!r} start={start!r}"
                f"end={end!r} interval={interval!r}. Check the symbol and "
                f"date range."
            )
        return self._normalize(raw, symbol=symbol)

    @staticmethod
    def _normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = raw.reset_index()

        #yfinance names the index column "Date" for daily/weekly/monthly
        #intervals and "Datetime" for intrayday intervals.

        date_col = "Date" if "Date" in df.columns else "Datetime"

        rename_map ={
            date_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }

    
