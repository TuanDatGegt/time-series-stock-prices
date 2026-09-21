## src/training/registry.py
"""
Module: src/training/registry.py
Description: Local and MLflow Model Registry integration for Phase 17.
How it works:
    Provides `register_local` for writing local JSON metadata manifests and
    `MLflowRegistryService` for logging parameters, metrics, artifacts, and managing
    model state transitions (Candidate -&gt; Production) in MLflow.
"""

from __future__ import annotations
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def register_local(
    metadata_dir: str | Path,
    symbol: str,
    model_name: str,
    metadata: Dict[str, Any],
) -> Path:
    """Write a local model registration JSON manifest [4]."""
    directory = Path(metadata_dir) / symbol.upper()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_name}.json"
    document = {
        "registration_version": 1,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "model_name": model_name,
        **metadata,
    }
    temporary = path.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, default=str)
        temporary.replace(path)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise OSError(f"Could not register model metadata at {path}: {exc}") from exc
    return path


class MLflowRegistryService:
    """
    Manages experiment tracking, artifact logging, and model registry via MLflow [1, 2].
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize MLflow client with target tracking server URI [5].
        """
        self.tracking_uri = tracking_uri
        self._mlflow = None

    def _get_mlflow(self):
        """Lazy load mlflow library with import guard."""
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                self._mlflow = mlflow
            except ImportError as exc:
                raise ImportError(
                    "mlflow package is required for MLflowRegistryService. "
                    "Install it via `pip install mlflow`."
                ) from exc
        return self._mlflow

    def _load_legacy_model(
        self, checkpoint_path: str | Path, metadata_path: str | Path | None = None
    ):
        checkpoint = Path(checkpoint_path)
        metadata = {}
        if checkpoint.suffix == ".json":
            with checkpoint.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            checkpoint = Path(metadata["checkpoint_path"])
            metadata_path = checkpoint_path
        elif metadata_path is not None:
            with Path(metadata_path).open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        model_name = str(metadata.get("model_name", "")).lower()
        if model_name in {"lstm", "gru"}:
            import torch

            from src.models.factory import create_model

            model, _ = create_model(
                model_name,
                num_features=len(metadata["feature_names"]),
                model_config=metadata.get("model_config", {}),
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload.get("model_state_dict", payload))
            model.eval()
            return model

        if checkpoint.suffix == ".pt":
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "model" in payload:
                return payload["model"]
            raise ValueError(
                "Legacy PyTorch checkpoint requires metadata with model_name and model_config"
            )

        with checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        return payload.get("model", payload)

    def _log_model_flavor(self, model, model_name: str) -> None:
        mlflow = self._get_mlflow()
        if model_name.lower() in {"lstm", "gru", "pytorch"}:
            mlflow.pytorch.log_model(
                model, artifact_path="model", serialization_format="pickle"
            )
        elif model_name.lower() == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

    def register_model(
        self,
        model,
        symbol: str,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Log a model flavor and return its standard MLflow runs URI."""
        mlflow = self._get_mlflow()
        mlflow.set_experiment(f"{symbol.upper()}-forecasting")
        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}"
        ) as run:
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.set_tag("symbol", symbol.upper())
            mlflow.set_tag("model_name", model_name)
            self._log_model_flavor(model, model_name)
            return f"runs:/{run.info.run_id}/model"

    def log_experiment_run(
        self,
        symbol: str,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Log hyperparameters, validation/test metrics, and checkpoint artifacts to MLflow [1].

        Args:
            symbol (str): Target stock ticker (e.g., 'INTC').
            model_name (str): Model architecture name ('lstm', 'gru', 'xgboost').
            params (Dict[str, Any]): Training hyperparameters (lookback, hidden_size, lr, etc.).
            metrics (Dict[str, float]): Evaluation metrics (MAE, RMSE, MAPE, R2, Direction Accuracy).
            artifacts (Optional[Dict[str, str]]): Paths to checkpoint files and scaler state dicts.
            tags (Optional[Dict[str, str]]): Optional Git commit hash or run tags.

        Returns:
            str: Unique MLflow Run ID.
        """
        mlflow = self._get_mlflow()
        experiment_name = f"{symbol.upper()}-forecasting"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            # Log hyperparameters
            mlflow.log_params(params)

            # Log metrics (MAE, RMSE, MAPE, R2, Directional Accuracy) [1]
            mlflow.log_metrics(metrics)

            # Log tags
            mlflow.set_tag("symbol", symbol.upper())
            mlflow.set_tag("model_name", model_name)
            if tags:
                mlflow.set_tags(tags)

            checkpoint_path = (artifacts or {}).get("checkpoint")
            metadata_path = (artifacts or {}).get("metadata")
            if checkpoint_path:
                model = self._load_legacy_model(checkpoint_path, metadata_path)
                self._log_model_flavor(model, model_name)

            # Log artifact files (checkpoints, scalers, metadata)
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path, artifact_path="artifacts")

            return run.info.run_id

    def promote_to_production(
        self,
        symbol: str,
        model_name: str,
        run_id: str,
        current_metrics: Dict[str, float],
        metric_key: str = "rmse",
    ) -> bool:
        """
        Promote candidate model to Production if it passes the Quality Gate check [3].

        Args:
            symbol (str): Target ticker symbol.
            model_name (str): Candidate model name.
            run_id (str): MLflow run ID to evaluate.
            current_metrics (Dict[str, float]): Current run's evaluation metrics.
            metric_key (str): Primary metric for Quality Gate comparison (default: 'rmse').

        Returns:
            bool: True if model passed quality gate and was promoted, False otherwise.
        """
        mlflow = self._get_mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        registered_model_name = f"{symbol.upper()}_{model_name.upper()}"

        # Register model candidate
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, registered_model_name)

        # Retrieve current Production model metrics for Quality Gate check
        try:
            prod_versions = client.get_latest_versions(
                registered_model_name, stages=["Production"]
            )
            if prod_versions:
                prod_run = client.get_run(prod_versions.run_id)
                prod_rmse = prod_run.data.metrics.get(metric_key, float("inf"))

                # Quality Gate Check: Reject if new model RMSE is not better [3]
                if current_metrics.get(metric_key, float("inf")) >= prod_rmse:
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=model_version.version,
                        stage="Archived",
                    )
                    return False
        except Exception:
            # If no production model exists yet, promote directly
            pass

        # Promote to Production stage
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_version.version,
            stage="Production",
        )
        return True

    def load_model(
        self,
        model_uri: str | Path,
        model_name: str | None = None,
        metadata_path: str | Path | None = None,
    ):
        """Load a flavor URI, with fallback support for legacy local artifacts."""
        uri = str(model_uri)
        if uri.startswith("runs:/") or uri.startswith("models:/"):
            mlflow = self._get_mlflow()
            name = (model_name or "pytorch").lower()
            if name in {"lstm", "gru", "pytorch"}:
                return mlflow.pytorch.load_model(uri)
            if name == "xgboost":
                return mlflow.xgboost.load_model(uri)
            return mlflow.sklearn.load_model(uri)
        return self._load_legacy_model(uri, metadata_path)
