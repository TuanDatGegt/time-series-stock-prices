## dashboard/services.py
"""
Module: dashboard/services.py
Description: HTTP REST Client Service for Streamlit Dashboard.
How it works:
    Communicates with FastAPI backend endpoints (`/health`, `/ready`, `/api/market`, `/api/prediction`).
    Handles network errors gracefully and parses JSON responses into pandas DataFrames and dictionaries.
"""

from typing import Any, Dict, Optional
import pandas as pd
import requests


class APIClient:
    """
    HTTP Client handling requests to the FastAPI forecasting backend.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API Client with base URL.
        """
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """
        Query GET /health and GET /ready endpoints.
        """
        try:
            resp = requests.get(f"{self.base_url}/ready", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "not_ready", "database": False, "model": False}
        except Exception:
            return {"status": "offline", "database": False, "model": False}

    def fetch_market_history(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars via GET /api/market/{symbol}/history.
        """
        url = f"{self.base_url}/api/market/{symbol.upper()}/history"
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch market data: {response.json().get('detail', 'Unknown error')}"
            )

        data = response.json()
        bars = data.get("bars", [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_prediction(self, symbol: str, model: str = "lstm") -> Dict[str, Any]:
        """
        Fetch next-day price forecast via GET /api/prediction/{symbol}.
        """
        url = f"{self.base_url}/api/prediction/{symbol.upper()}"
        params = {"model": model.lower()}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch prediction: {response.json().get('detail', 'Unknown error')}"
            )

        return response.json()
