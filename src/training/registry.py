from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def register_local(
    metadata_dir: str | Path,
    symbol: str,
    model_name: str,
    metadata: dict[str, Any],
) -> Path:
    """Write a future-MLflow-compatible local model registration manifest."""
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