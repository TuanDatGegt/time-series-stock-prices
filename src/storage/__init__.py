# src/storage/__init__.py
"""
Module: src/storage/__init__.py
Description: Storage package initializer.
How it works:
    Exposes primary engine creation helpers (`init_db`, `create_engine`),
    database models (`MarketData`), and repository class (`MarketDataRepository`)
    at the package level for clean imports throughout the application.
"""

from src.storage.database import Base, SessionLocal, create_engine, init_db
from src.storage.models import MarketData
from src.storage.repository import MarketDataRepository

__all__ = [
    "Base",
    "SessionLocal",
    "create_engine",
    "init_db",
    "MarketData",
    "MarketDataRepository",
]
