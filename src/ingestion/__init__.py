## src/ingestion/__init__.py

"""
Module: src/ingestion/__init__.py
Description: Package initializer for `src.ingestion`.
How it works:
    Exposes key interfaces and service classes at the package level for clean imports across the project.
"""

from src.ingestion.base import MarketDataSource
from src.ingestion.service import IncrementalIngestionService
from src.ingestion.yahoo import YahooFinanceSource

__all__ = [
    "MarketDataSource",
    "YahooFinanceSource",
    "IncrementalIngestionService",
]
