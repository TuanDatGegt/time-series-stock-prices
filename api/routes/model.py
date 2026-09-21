from fastapi import APIRouter, Depends, HTTPException

from api.routes.prediction import get_inference_service
from src.inference.service import InferenceService

router = APIRouter(tags=["Model"])


@router.get("/api/model/{symbol}")
def get_model_metadata(
    symbol: str,
    service: InferenceService = Depends(get_inference_service),
):
    """Return registered model metadata for a symbol."""
    try:
        predictor = service.get_predictor(symbol.upper())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return predictor.metadata


@router.get("/api/metrics/{symbol}")
def get_model_metrics(
    symbol: str,
    service: InferenceService = Depends(get_inference_service),
):
    """Return registered validation and test metrics for a symbol."""
    try:
        predictor = service.get_predictor(symbol.upper())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    metadata = predictor.metadata
    metrics = {
        "symbol": symbol.upper(),
        "model_name": metadata.get("model_name"),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "test_metrics": metadata.get("test_metrics", {}),
    }
    if not metrics["validation_metrics"] and not metrics["test_metrics"]:
        raise HTTPException(
            status_code=404, detail=f"No metrics found for symbol {symbol}"
        )
    return metrics
