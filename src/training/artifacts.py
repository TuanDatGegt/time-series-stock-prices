## src/training/artifacts.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = 1


def save_artifact(
    path: str | Path,
    model_name: str,
    model,
    metadata: dict[str, Any],
) -> Path:
    """Persist a versioned Phase 13 artifact and its reproducibility metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        **metadata,
        "model_name": model_name,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        if model_name in {"lstm", "gru"}:
            import torch

            payload["model_state_dict"] = model.state_dict()
            torch.save(payload, temporary)
        else:
            payload["model"] = model
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle)
        temporary.replace(destination)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise OSError(f"Could not save checkpoint to {destination}: {exc}") from exc
    return destination
