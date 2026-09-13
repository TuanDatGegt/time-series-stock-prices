## src/ingestion/base.py

"""
Module: src/ingestion/base.py
Description: Defines the abstract base class (interface) for all market data sources.
How it works:
    This file establishes a uniform Data Contract across different data vendors (e.g., Yahoo Finance, Alpha Vantage).
    Any custom market data provider must inherit from `MarketDataSource` and implement its abstract methods to
    fetch historical and latest real-time/near real-time records into a standardized OHLCV schema.
"""

from abc import ABC, abstractmethod
import pandas as pd


class MarketDataSource(ABC):
    """
    Abstract interface for market data providers.
    Enforces standardized data schema: symbol, timestamp, open, high, low, close, volume.
    """

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical market OHLCV data within a given date range.

        Args:
            symbol (str): Target ticker symbol (e.g., 'INTC').
            start (str): Start date string (YYYY-MM-DD or ISO timestamp).
            end (str): End date string (YYYY-MM-DD or ISO timestamp).
            interval (str): Data sampling frequency (e.g., '1d', '1h', '1m').

        Returns:
            pd.DataFrame: Standardized DataFrame containing [symbol, timestamp, open, high, low, close, volume].
        """

        pass

    @abstractmethod
    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """
        Fetch the most recent single candle/tick record for a specific symbol.

        Args:
            symbol (str): Target ticker symbol.

        Returns:
            pd.DataFrame: Single-row DataFrame matching the standardized schema.
        """

    def fetch_latest_price(self, symbol: str) -> float:
        """
        Fetch the latest closing price for a given symbol.

        Args:
           symbol (str): Target ticker symbol.
        Returns:
           float: Latest closing price.
        """
        return self.fetch_latest(symbol)["close"].iloc[0]
