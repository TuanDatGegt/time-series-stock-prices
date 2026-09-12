### **src/storage/models.py**

"""
Module: src/storage/models.py
Description: SQLAlchemy ORM Models for Database Tables.
How it works:
    Defines database table schemas (`market_data`) mapped to Python objects.
    Enforces data integrity through column constraints, primary keys, auto-increment IDs,
    and unique constraints on `(symbol, timestamp)`.
"""

from __future__ import annotations
from sqlalchemy import Column, Float, Integer, String, TIMESTAMP, UniqueConstraint
from src.storage.database import Base


class MarketData(Base):
    """
    ORM Model for storing market OHLCV price records.
    Mapped to table: `market_data`.
    """

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_symbol_timestamp"),
        {"sqlite_autoincrement": True},
    )
