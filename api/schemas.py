## api/schemas.py
"""
Module: api/schemas.py
Description: Pydantic Data Contracts and Response Schemas for Phase 20 API endpoints.
How it works:
    Defines request/response validation schemas for prediction results, market bar history,
    model metadata status, and system health checks to enforce strict API response shapes.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Standardized payload schema for price prediction endpoint response.
    """

    symbol: str = Field(..., example="INTC", description="Stock ticker symbol")
    timestamp: str = Field(
        ..., example="2026-09-14 00:00:00", description="Latest bar timestamp"
    )
    current_price: float = Field(..., example=25.43, description="Latest closing price")
    predicted_price: float = Field(
        ..., example=26.02, description="Forecasted next-day closing price"
    )
    price_change: float = Field(
        ..., example=0.59, description="Absolute price difference"
    )
    predicted_change_pct: float = Field(
        ..., example=2.32, description="Expected percentage change"
    )
    direction: str = Field(
        ..., example="UP", description="Predicted trend direction (UP/DOWN)"
    )
    model_name: str = Field(
        ..., example="lstm", description="Forecasting model architecture"
    )
    model_version: int = Field(..., example=1, description="Model registration version")


class MarketBarSchema(BaseModel):
    """
    OHLCV Market Data bar schema.
    """

    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketHistoryResponse(BaseModel):
    """
    Response payload schema for historical market bars query.
    """

    symbol: str
    count: int
    bars: List[MarketBarSchema]


class HealthCheckResponse(BaseModel):
    """
    Simple liveness check schema.
    """

    status: str = Field(default="ok", example="ok")


class ReadinessCheckResponse(BaseModel):
    """
    Readiness check schema evaluating database and model artifact availability.
    """

    status: str = Field(..., example="ready")
    database: bool = Field(..., example=True)
    model: bool = Field(..., example=True)
