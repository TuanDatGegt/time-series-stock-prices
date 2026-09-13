## src/storage/repository.py

"""
Module: src/storage/repository.py
Description: Data Access Object (DAO) / Repository pattern for database operations.
How it works:
    Interacts with `MarketData` tables using SQLAlchemy ORM sessions.
    Handles batch UPSERT logic (insert new or update modified records), queries historical OHLCV data,
    and retrieves the latest recorded timestamp (`get_last_timestamp`) to support incremental ingestion.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.storage.models import MarketData


class MarketDataRepository:
    """
    Repository class handling CRUD operations for MarketData records.
    """

    def __init__(self, engine: Engine):
        """
        Initialize repository with a database Engine.
        """
        self.engine = engine

    def upsert_batch(self, df: pd.DataFrame) -> int:
        """
        Batch UPSERT market data records into the database.
        Inserts non-existent records and updates existing records if values change.

        Args:
            df (pd.DataFrame): Validated DataFrame containing [symbol, timestamp, open, high, low, close, volume].

        Returns:
            int: Number of records inserted or updated.
        """
        if df is None or df.empty:
            return 0

        records = df.copy()
        records["timestamp"] = pd.to_datetime(records["timestamp"])
        records = records[
            ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        ].copy()

        with Session(self.engine) as session:
            changed_count = 0
            for _, row in records.iterrows():
                symbol = str(row["symbol"])
                timestamp = row["timestamp"]

                # Query existing record matching unique key (symbol, timestamp)
                existing = session.execute(
                    select(MarketData).where(
                        and_(
                            MarketData.symbol == symbol,
                            MarketData.timestamp == timestamp,
                        )
                    )
                ).scalar_one_or_none()

                # Insert new record if it does not exist
                if existing is None:
                    session.add(
                        MarketData(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=int(row["volume"]),
                        )
                    )
                    changed_count += 1
                    continue

                # Update existing record if any OHLCV value differs
                new_values = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
                if (
                    existing.open != new_values["open"]
                    or existing.high != new_values["high"]
                    or existing.low != new_values["low"]
                    or existing.close != new_values["close"]
                    or existing.volume != new_values["volume"]
                ):
                    existing.open = new_values["open"]
                    existing.high = new_values["high"]
                    existing.low = new_values["low"]
                    existing.close = new_values["close"]
                    existing.volume = new_values["volume"]
                    changed_count += 1

            session.commit()
            return changed_count

    def fetch_history(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical market records for a given symbol within optional date boundaries.

        Args:
            symbol (str): Target stock ticker.
            start (Optional[str]): Start date bound.
            end (Optional[str]): End date bound.

        Returns:
            pd.DataFrame: DataFrame containing fetched records ordered chronologically.
        """
        with Session(self.engine) as session:
            query = select(MarketData).where(MarketData.symbol == symbol)
            if start is not None:
                query = query.where(MarketData.timestamp >= pd.to_datetime(start))
            if end is not None:
                query = query.where(MarketData.timestamp <= pd.to_datetime(end))

            query = query.order_by(MarketData.timestamp.asc())
            rows = session.execute(query).scalars().all()

        if not rows:
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

        return pd.DataFrame(
            [
                {
                    "symbol": row.symbol,
                    "timestamp": row.timestamp,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                }
                for row in rows
            ]
        )

    def get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Retrieve the latest recorded timestamp for a symbol to support incremental data fetching.

        Args:
            symbol (str): Target stock ticker.

        Returns:
            Optional[datetime]: Max timestamp datetime object or None if no records exist.
        """
        with Session(self.engine) as session:
            row = session.execute(
                select(MarketData.timestamp)
                .where(MarketData.symbol == symbol)
                .order_by(MarketData.timestamp.desc())
                .limit(1)
            ).scalar_one_or_none()
            return row
