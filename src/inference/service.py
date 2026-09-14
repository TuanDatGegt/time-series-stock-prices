## src/inference/service.py
"""
Module: src/inference/service.py
Description: High-Level Inference Service with Model Instance Caching.
How it works:
    Serves as the primary entry point for FastAPI and Streamlit dashboard requests.
    Caches loaded `Predictor` instances in memory (`_cached_predictors`) to avoid expensive
    disk reads on every request, fetches fresh market history from `MarketDataRepository`,
    and generates price forecasts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from src.inference.predictor import Predictor
from src.storage.repository import MarketDataRepository


class InferenceService:
    """
    Facade service orchestrating predictor initialization, caching, and repository data fetching.
    """

    def __init__(
        self,
        repository: MarketDataRepository,
        checkpoint_dir: str | Path = "models/checkpoints",
        metadata_dir: str | Path = "models/metadata",
    ):
        """
        Initialize InferenceService.

        Args:
            repository (MarketDataRepository): Database repository for fetching price history.
            checkpoint_dir (str | Path): Root directory storing checkpoints.
            metadata_dir (str | Path): Root directory storing metadata JSON files.
        """
        self.repository = repository
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metadata_dir = Path(metadata_dir)
        self._cached_predictors: Dict[str, Predictor] = {}

    def get_predictor(self, symbol: str, model_name: str = "lstm") -> Predictor:
        """
        Retrieve a cached Predictor instance or instantiate a new one from disk.

        Args:
            symbol (str): Target stock ticker (e.g., 'INTC').
            model_name (str): Architecture name ('lstm', 'gru', 'xgboost').

        Returns:
            Predictor: Loaded and cached Predictor instance.
        """
        cache_key = f"{symbol.upper()}_{model_name.lower()}"
        if cache_key in self._cached_predictors:
            return self._cached_predictors[cache_key]

        symbol_clean = symbol.upper()
        model_clean = model_name.lower()

        metadata_path = self.metadata_dir / symbol_clean / f"{model_clean}.json"

        # Determine checkpoint filename extension
        ext = ".pt" if model_clean in {"lstm", "gru"} else ".pkl"
        checkpoint_path = (
            self.checkpoint_dir / symbol_clean / f"{model_clean}_best{ext}"
        )
        if not checkpoint_path.exists():
            checkpoint_path = self.checkpoint_dir / symbol_clean / f"{model_clean}{ext}"

        predictor = Predictor(
            checkpoint_path=checkpoint_path, metadata_path=metadata_path
        )
        self._cached_predictors[cache_key] = predictor
        return predictor

    def predict_symbol(self, symbol: str, model_name: str = "lstm") -> Dict[str, Any]:
        """
        Fetch latest market data for symbol and generate next-day price forecast.

        Args:
            symbol (str): Target stock ticker.
            model_name (str): Candidate model name.

        Returns:
            Dict[str, Any]: Standardized prediction output dictionary.
        """
        predictor = self.get_predictor(symbol, model_name)

        # Fetch last 200 bars to ensure sufficient history for technical indicator warmup
        history_df = self.repository.fetch_history(symbol=symbol.upper())
        if history_df.empty:
            raise ValueError(
                f"No market records found in repository for symbol: {symbol}"
            )

        return predictor.predict_next(history_df)
