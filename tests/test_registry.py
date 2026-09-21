# tests/test_registry.py

import numpy as np
import torch
import json

from src.training.registry import MLflowRegistryService


def test_register_and_load_pytorch_model_flavor(tmp_path):
    service = MLflowRegistryService(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    model = torch.nn.Linear(2, 1)
    model.eval()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = model(inputs).detach().numpy()

    model_uri = service.register_model(
        model=model,
        symbol="INTC",
        model_name="pytorch",
    )
    loaded = service.load_model(model_uri, model_name="pytorch")
    actual = loaded(inputs).detach().numpy()

    assert model_uri.startswith("runs:/")
    np.testing.assert_allclose(actual, expected)


def test_load_model_falls_back_to_legacy_pt_json_artifacts(tmp_path):
    service = MLflowRegistryService(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "legacy.pt"
    metadata_path = tmp_path / "legacy.json"
    torch.save({"model": model}, checkpoint_path)
    metadata_path.write_text(
        json.dumps({"checkpoint_path": str(checkpoint_path), "model_name": "pytorch"}),
        encoding="utf-8",
    )
    inputs = torch.tensor([[1.0, 2.0]])

    loaded = service.load_model(metadata_path)

    np.testing.assert_allclose(
        loaded(inputs).detach().numpy(), model(inputs).detach().numpy()
    )
