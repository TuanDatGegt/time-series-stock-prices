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
        model_uri = f"runs:/{run_id}/artifacts"
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
