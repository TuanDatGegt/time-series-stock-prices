## api/routes/prediction.py
"""
Module: api/routes/prediction.py
Description: Router handling model prediction endpoints.
How it works:
    Exposes `GET /api/prediction/{symbol}`. Invokes `InferenceService` to load
    cached predictor models, fetch recent market history, and calculate predictions.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from api.schemas import PredictionResponse
from src.inference.service import InferenceService

router = APIRouter(prefix="/api/prediction", tags=["Prediction"])


def get_inference_service() -> InferenceService:
    """
    Dependency provider for InferenceService instance (configured in app lifespan).
    """
    from api.main import app_state

    if app_state.inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service unavailable")
    return app_state.inference_service


@router.get("/{symbol}", response_model=PredictionResponse)
def get_symbol_prediction(
    symbol: str,
    model: str = Query(
        "lstm", description="Target model architecture (lstm, gru, xgboost)"
    ),
    service: InferenceService = Depends(get_inference_service),
):
    """
    Generate next-day price forecast for a given stock symbol.
    """
    try:
        result = service.predict_symbol(symbol=symbol.upper(), model_name=model.lower())
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
