## api/routes/market.py
"""
Module: api/routes/market.py
Description: Router handling market data queries.
How it works:
    Exposes `GET /api/market/{symbol}/history` to retrieve historical OHLCV records from SQL storage.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from api.schemas import MarketBarSchema, MarketHistoryResponse
from src.storage.repository import MarketDataRepository

router = APIRouter(prefix="/api/market", tags=["Market Data"])


def get_repository() -> MarketDataRepository:
    """
    Dependency provider for MarketDataRepository.
    """
    from api.main import app_state

    if app_state.repository is None:
        raise HTTPException(status_code=503, detail="Database repository unavailable")
    return app_state.repository


@router.get("/{symbol}/history", response_model=MarketHistoryResponse)
def get_market_history(
    symbol: str,
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    repository: MarketDataRepository = Depends(get_repository),
):
    """
    Fetch historical OHLCV bars for a specified symbol and date range.
    """
    df = repository.fetch_history(symbol=symbol.upper(), start=start, end=end)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No market data found for symbol {symbol}"
        )

    bars = [
        MarketBarSchema(
            symbol=row["symbol"],
            timestamp=str(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
        )
        for _, row in df.iterrows()
    ]

    return MarketHistoryResponse(symbol=symbol.upper(), count=len(bars), bars=bars)


@router.get("/{symbol}", response_model=MarketBarSchema)
def get_latest_market_data(
    symbol: str,
    repository: MarketDataRepository = Depends(get_repository),
):
    """Fetch the latest stored market bar for a symbol."""
    df = repository.fetch_history(symbol=symbol.upper())
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No market data found for symbol {symbol}"
        )

    row = df.iloc[-1]
    return MarketBarSchema(
        symbol=row["symbol"],
        timestamp=str(row["timestamp"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=int(row["volume"]),
    )
