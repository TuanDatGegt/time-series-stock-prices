## src/training/train.py

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.features.builder import build_features
from src.models.factory import create_model
from src.preprocessing.scaler import FeatureScaler
from src.preprocessing.sequence import build_sequences
from src.preprocessing.splitter import split_dataset
from src.storage.database import create_engine
from src.storage.repository import MarketDataRepository
from src.training.artifacts import save_artifact
from src.training.evaluate import evaluate_arrays, predict_torch
from src.training.early_stopping import EarlyStopping
from src.training.registry import register_local
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.validation.market_data import MarketDataValidator


@dataclass
class TrainingResult:
    symbol: str
    model_name: str
    checkpoint_path: str
    metadata_path: str
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    history: list[dict[str, float]]


def _as_metadata(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _resolve_device(config: dict[str, Any]) -> str:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
    return requested


def _previous_close(frame: pd.DataFrame, lookback: int, horizon: int) -> np.ndarray:
    start = lookback + horizon - 2
    end = start + len(frame) - lookback - horizon + 1
    return frame["close"].to_numpy(dtype=np.float64)[start:end]


def _validate_sequence_set(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    feature_count: int,
) -> None:
    if X.ndim != 3 or X.shape[1:] != (lookback, feature_count):
        raise ValueError(
            f"Incompatible sequence dimensions: expected (*, {lookback}, "
            f"{feature_count}), got {X.shape}"
        )
    if y.shape != (len(X), 1):
        raise ValueError(
            f"Incompatible target dimensions: expected ({len(X)}, 1), got {y.shape}"
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Sequences contain non-finite values")


def _train_torch(
    model,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    config,
    device: str,
    patience: int,
    min_delta: float,
) -> tuple[list[dict[str, float]], dict[str, Any], int]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    val_features = torch.from_numpy(val_X).to(device=device, dtype=torch.float32)
    val_targets = torch.from_numpy(val_y).to(device=device, dtype=torch.float32)
    history: list[dict[str, float]] = []
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(val_features), val_targets).cpu())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        result = early_stopping.step(validation_loss, epoch)
        if result.improved:
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            best_epoch = epoch
        if result.should_stop:
            break

    if best_state is None:
        raise RuntimeError("EarlyStopping did not produce a best model state")
    return history, best_state, best_epoch


def run_training(
    symbol: str,
    model_name: str,
    config: dict[str, Any] | None = None,
    repository: MarketDataRepository | None = None,
    data_loader: Callable[[str], pd.DataFrame] | None = None,
    checkpoint_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    logger=None,
) -> TrainingResult:
    """Execute the fixed-epoch Phase 13 training flow."""
    if not symbol or not symbol.strip():
        raise ValueError("symbol must not be empty")
    symbol = symbol.strip().upper()
    model_name = model_name.strip().lower()
    resolved = config or load_config()
    logger = logger or get_logger(
        level=resolved.get("logging", {}).get("level", "INFO")
    )
    training = resolved["training"]
    feature_cols = list(training["feature_columns"])
    target_col = training["target_column"]
    timestamp_col = resolved.get("data", {}).get("timestamp_column", "timestamp")

    if data_loader is None:
        if repository is None:
            database_url = resolved.get("database_url")
            if not database_url:
                raise ValueError("database_url is required to load training data")
            repository = MarketDataRepository(create_engine(database_url))
        data_loader = repository.fetch_history
    try:
        raw = data_loader(symbol)
    except Exception as exc:
        raise RuntimeError(f"Could not load data for {symbol}: {exc}") from exc
    if raw is None or raw.empty:
        raise ValueError(f"No market data found for symbol {symbol}")

    valid, invalid = MarketDataValidator().validate(raw)
    if invalid is not None and not invalid.empty:
        logger.warning("Dropped %d invalid market rows for %s", len(invalid), symbol)
    if valid.empty:
        raise ValueError(f"No valid market data remains for symbol {symbol}")
    valid = valid[valid["symbol"].str.upper() == symbol].copy()
    if valid.empty:
        raise ValueError(f"No validated rows found for symbol {symbol}")

    featured = build_features(valid)
    missing = [
        column
        for column in feature_cols + [target_col]
        if column not in featured.columns
    ]
    if missing:
        raise ValueError(
            f"Configured columns missing after feature creation: {missing}"
        )
    featured = featured.sort_values(timestamp_col).reset_index(drop=True)
    featured = featured.dropna(subset=feature_cols + [target_col]).reset_index(
        drop=True
    )
    if featured.empty:
        raise ValueError("Feature warmup removed all usable rows")

    train_df, val_df, test_df = split_dataset(
        featured,
        split_mode=training.get("split_mode", "ratio"),
        timestamp_col=timestamp_col,
        train_end=training.get("train_end"),
        val_end=training.get("val_end"),
        train_ratio=training.get("train_ratio", 0.7),
        val_ratio=training.get("val_ratio", 0.15),
    )
    scaler = FeatureScaler().fit(train_df[feature_cols])
    scaled_frames = []
    for frame in (train_df, val_df, test_df):
        scaled = frame.copy()
        scaled.loc[:, feature_cols] = scaler.transform(frame[feature_cols]).to_numpy()
        scaled_frames.append(scaled)
    scaled_train, scaled_val, scaled_test = scaled_frames
    logger.info(
        "Dataset split: symbol=%s train=%d validation=%d test=%d features=%d",
        symbol,
        len(train_df),
        len(val_df),
        len(test_df),
        len(feature_cols),
    )

    model_config = resolved.get("models", {}).get(model_name)
    if model_config is None:
        raise ValueError(f"Missing configuration for model '{model_name}'")
    lookback = int(model_config.get("lookback", training["lookback"]))
    horizon = int(model_config.get("horizon", training["horizon"]))
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")

    history: list[dict[str, float]] = []
    if model_name in {"lstm", "gru"}:
        train_X, train_y = build_sequences(
            scaled_train, lookback, horizon, feature_cols, target_col
        )
        val_X, val_y = build_sequences(
            scaled_val, lookback, horizon, feature_cols, target_col
        )
        test_X, test_y = build_sequences(
            scaled_test, lookback, horizon, feature_cols, target_col
        )
        for X, y in ((train_X, train_y), (val_X, val_y), (test_X, test_y)):
            _validate_sequence_set(X, y, lookback, len(feature_cols))
        logger.info(
            "Sequence counts: train=%d validation=%d test=%d",
            len(train_X),
            len(val_X),
            len(test_X),
        )
        model, model_config_object = create_model(
            model_name, len(feature_cols), model_config
        )
        device = _resolve_device(resolved)
        seed = int(resolved.get("random_seed", 42))
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        patience = int(model_config.get("patience", training.get("patience", 10)))
        min_delta = float(model_config.get("min_delta", training.get("min_delta", 0.0)))
        if patience < 0 or min_delta < 0:
            raise ValueError(
                "early stopping patience must be >= 0 and min_delta must be >= 0"
            )
        history, best_state, best_epoch = _train_torch(
            model,
            train_X,
            train_y,
            val_X,
            val_y,
            model_config_object,
            device,
            patience,
            min_delta,
        )
        model.load_state_dict(best_state)
        logger.info(
            "Early stopping selected epoch %d after %d training epochs",
            best_epoch,
            len(history),
        )
        val_pred = predict_torch(model, val_X, device)
        test_pred = predict_torch(model, test_X, device)
        val_true = val_y.reshape(-1)
        test_true = test_y.reshape(-1)
        val_previous = _previous_close(val_df, lookback, horizon)
        test_previous = _previous_close(test_df, lookback, horizon)
    elif model_name == "xgboost":
        from src.models.baseline import make_target

        train_tabular = make_target(scaled_train).dropna(subset=["target_close_next"])
        val_tabular = make_target(scaled_val).dropna(subset=["target_close_next"])
        test_tabular = make_target(scaled_test).dropna(subset=["target_close_next"])
        if min(len(train_tabular), len(val_tabular), len(test_tabular)) < 1:
            raise ValueError("Insufficient samples for xgboost training/evaluation")
        model, model_config_object = create_model(
            model_name, len(feature_cols), model_config, feature_cols=feature_cols
        )
        model.fit(train_tabular)
        val_pred = model.predict(val_tabular)
        test_pred = model.predict(test_tabular)
        val_true = val_tabular["target_close_next"].to_numpy()
        test_true = test_tabular["target_close_next"].to_numpy()
        val_previous = val_tabular[target_col].to_numpy()
        test_previous = test_tabular[target_col].to_numpy()
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    validation_metrics = evaluate_arrays(val_previous, val_true, val_pred, model_name)
    test_metrics = evaluate_arrays(test_previous, test_true, test_pred, model_name)
    artifact_metadata = {
        "symbol": symbol,
        "feature_names": feature_cols,
        "target": target_col,
        "lookback": lookback,
        "horizon": horizon,
        "model_config": _as_metadata(model_config_object),
        "scaler": scaler.state_dict(),
        "split_metadata": {
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "train_start": str(train_df[timestamp_col].min()),
            "train_end": str(train_df[timestamp_col].max()),
            "validation_start": str(val_df[timestamp_col].min()),
            "validation_end": str(val_df[timestamp_col].max()),
            "test_start": str(test_df[timestamp_col].min()),
            "test_end": str(test_df[timestamp_col].max()),
        },
        "sequence_counts": {
            "train": len(train_X) if model_name in {"lstm", "gru"} else None,
            "validation": len(val_X) if model_name in {"lstm", "gru"} else None,
            "test": len(test_X) if model_name in {"lstm", "gru"} else None,
        },
        "history": history,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "device": device if model_name in {"lstm", "gru"} else "cpu",
        "best_epoch": best_epoch if model_name in {"lstm", "gru"} else None,
    }
    checkpoint_root = Path(checkpoint_dir or resolved["paths"]["checkpoint_dir"])
    suffix = ".pt" if model_name in {"lstm", "gru"} else ".pkl"
    best_filename = (
        f"{model_name}_best.pt"
        if model_name in {"lstm", "gru"}
        else f"{model_name}_best.pkl"
    )
    checkpoint_path = save_artifact(
        checkpoint_root / symbol / best_filename,
        model_name,
        model,
        artifact_metadata,
    )
    # Keep the Phase 13 path as a compatibility alias; both artifacts contain
    # the restored best state, while the best-named path is canonical.
    save_artifact(
        checkpoint_root / symbol / f"{model_name}{suffix}",
        model_name,
        model,
        artifact_metadata,
    )
    metadata_root = Path(metadata_dir or resolved["paths"]["metadata_dir"])
    metadata_path = register_local(
        metadata_root,
        symbol,
        model_name,
        {"checkpoint_path": str(checkpoint_path), **artifact_metadata},
    )

    # --- Phase 17: MLflow Model Registry Integration ---
    try:
        from src.training.registry import MLflowRegistryService

        mlflow_service = MLflowRegistryService(
            tracking_uri=resolved.get("mlflow_tracking_uri", "http://localhost:5000")
        )

        # 1. Log experiment run parameters &amp; metrics [1]
        run_id = mlflow_service.log_experiment_run(
            symbol=symbol,
            model_name=model_name,
            params={
                "lookback": lookback,
                "horizon": horizon,
                "feature_columns": feature_cols,
                "device": device if model_name in {"lstm", "gru"} else "cpu",
                **(
                    _as_metadata(model_config_object)
                    if is_dataclass(model_config_object)
                    else {}
                ),
            },
            metrics={
                **test_metrics,  # MAE, RMSE, MAPE, R2, Directional Accuracy [7]
                "best_epoch": best_epoch if model_name in {"lstm", "gru"} else 0,
            },
            artifacts={
                "checkpoint": str(checkpoint_path),
                "metadata": str(metadata_path),
            },
        )

        # 2. Evaluate Quality Gate &amp; transition stage [3]
        is_promoted = mlflow_service.promote_to_production(
            symbol=symbol,
            model_name=model_name,
            run_id=run_id,
            current_metrics=test_metrics,
            metric_key="rmse",
        )
        logger.info(
            "MLflow logging complete. Run ID: %s | Promoted to Prod: %s",
            run_id,
            is_promoted,
        )

    except Exception as exc:
        logger.warning("MLflow logging skipped or failed: %s", exc)

    logger.info(
        "Training complete: symbol=%s model=%s checkpoint=%s",
        symbol,
        model_name,
        checkpoint_path,
    )
    return TrainingResult(
        symbol=symbol,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        metadata_path=str(metadata_path),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        history=history,
    )
