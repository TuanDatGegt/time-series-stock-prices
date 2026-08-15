from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.storage.database import MarketData


class MarketDataRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def upsert_batch(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0

        records = df.copy()
        records["timestamp"] = pd.to_datetime(records["timestamp"])
        records = records[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()

        with Session(self.engine) as session:
            changed_count = 0
            for _, row in records.iterrows():
                symbol = str(row["symbol"])
                timestamp = row["timestamp"]
                existing = session.execute(
                    select(MarketData).where(
                        and_(MarketData.symbol == symbol, MarketData.timestamp == timestamp)
                    )
                ).scalar_one_or_none()

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

    def fetch_history(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        with Session(self.engine) as session:
            query = select(MarketData).where(MarketData.symbol == symbol)
            if start is not None:
                query = query.where(MarketData.timestamp >= pd.to_datetime(start))
            if end is not None:
                query = query.where(MarketData.timestamp <= pd.to_datetime(end))

            query = query.order_by(MarketData.timestamp.asc())
            rows = session.execute(query).scalars().all()

        if not rows:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])

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
        with Session(self.engine) as session:
            row = (
                session.execute(
                    select(MarketData.timestamp)
                    .where(MarketData.symbol == symbol)
                    .order_by(MarketData.timestamp.desc())
                    .limit(1)
                )
                .scalar_one_or_none()
            )
            return row
