## src/inference/predictor.py
"""
Module: src/inference/predictor.py
Description: Model Predictor and Sequence Inference Handler for Phase 19.
How it works:
    Loads registered model checkpoints (.pt/.pkl) and metadata JSON manifests.
    Restores fitted `FeatureScaler` states without refitting to prevent feature drift.
    Computes technical features on input market history, extracts the latest lookback window,
    scales features, builds sequence tensors, and executes forward pass predictions.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import torch

from src.features.builder import build_features
from src.models.factory import create_model
from src.preprocessing.scaler import FeatureScaler


class Predictor:
    """
    Core inference engine responsible for loading trained model artifacts
    and executing single-step price forecasts on fresh market data.
    """

    def __init__(self, checkpoint_path: str | Path, metadata_path: str | Path):
        """
        Initialize predictor by loading metadata and model weights.

        Args:
            checkpoint_path (str | Path): Path to trained model checkpoint (.pt or .pkl).
            metadata_path (str | Path): Path to JSON metadata manifest.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )

        # 1. Load JSON metadata manifest
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.symbol = self.metadata["symbol"]
        self.model_name = self.metadata["model_name"].lower()
        self.lookback = int(self.metadata["lookback"])
        self.horizon = int(self.metadata["horizon"])
        self.feature_names = list(self.metadata["feature_names"])
        self.target_col = str(self.metadata.get("target", "close"))

        # 2. Restore FeatureScaler state without refitting
        scaler_state = self.metadata["scaler"]
        self.scaler = FeatureScaler()
        self.scaler.feature_names = list(scaler_state["feature_names"])
        self.scaler.mean = np.array(scaler_state["mean"], dtype=np.float64)
        self.scaler.scale = np.array(scaler_state["scale"], dtype=np.float64)

        # 3. Restore Model instance and load checkpoint state
        self.model = self._load_model_checkpoint()

    def _load_model_checkpoint(self) -> Any:
        """
        Instantiate model class and load saved weights.
        """
        model_config = self.metadata.get("model_config", {})

        if self.model_name in {"lstm", "gru"}:
            model_inst, _ = create_model(
                self.model_name,
                num_features=len(self.feature_names),
                model_config=model_config,
            )
            payload = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = payload.get("model_state_dict", payload)
            model_inst.load_state_dict(state_dict)
            model_inst.eval()
            return model_inst
        elif self.model_name == "xgboost":
            with self.checkpoint_path.open("rb") as f:
                payload = pickle.load(f)
            return payload.get("model", payload)
        else:
            raise ValueError(f"Unsupported inference model: {self.model_name}")

    def predict_next(self, raw_history_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute price prediction using raw market history DataFrame.

        Args:
            raw_history_df (pd.DataFrame): Raw market OHLCV bars (must have &gt;= lookback + 50 rows for warmup).

        Returns:
            Dict[str, Any]: Prediction payload containing symbol, predicted_price, change_pct, and direction.
        """
        if raw_history_df is None or raw_history_df.empty:
            raise ValueError("Input history DataFrame cannot be empty")

        # 1. Compute technical indicator features
        featured_df = build_features(raw_history_df)
        featured_df = featured_df.sort_values("timestamp").reset_index(drop=True)
        featured_df = featured_df.dropna(subset=self.feature_names + [self.target_col])

        if len(featured_df) < self.lookback:
            raise ValueError(
                f"Insufficient valid rows after feature calculation: "
                f"got {len(featured_df)}, required lookback={self.lookback}"
            )

        # Extract latest rows for lookback window
        latest_window = featured_df.iloc[-self.lookback :].copy()
        current_price = float(latest_window[self.target_col].iloc[-1])
        latest_timestamp = str(latest_window["timestamp"].iloc[-1])

        # 2. Scale feature window
        scaled_features = self.scaler.transform(latest_window[self.feature_names])

        # 3. Run Inference
        if self.model_name in {"lstm", "gru"}:
            # Shape tensor to (1, lookback, num_features)
            input_tensor = torch.from_numpy(
                scaled_features.to_numpy(dtype=np.float32)
            ).unsqueeze(0)

            with torch.no_grad():
                pred_scaled = float(self.model(input_tensor).numpy().reshape(-1))

            # Note: Prediction outputs the target price directly or scaled value
            predicted_price = pred_scaled
        else:
            # XGBoost / Tabular inference on last row
            last_row = scaled_features.iloc[[-1]].to_numpy()
            predicted_price = float(self.model.predict(last_row))

        price_change = predicted_price - current_price
        change_pct = (
            (price_change / current_price) * 100.0 if current_price != 0 else 0.0
        )
        direction = "UP" if price_change >= 0 else "DOWN"

        return {
            "symbol": self.symbol,
            "timestamp": latest_timestamp,
            "current_price": round(current_price, 4),
            "predicted_price": round(predicted_price, 4),
            "price_change": round(price_change, 4),
            "predicted_change_pct": round(change_pct, 2),
            "direction": direction,
            "model_name": self.model_name,
            "model_version": self.metadata.get("registration_version", 1),
        }
