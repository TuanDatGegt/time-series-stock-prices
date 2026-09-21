## api/main.py
"""
Module: api/main.py
Description: Main FastAPI Application Entrypoint and Health Check Endpoints.
How it works:
    Configures app lifecycle (database connection, service startup), mounts route handlers,
    and exposes `/health` and `/ready` endpoints for container orchestration health checks.
"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import market, model, prediction
from api.schemas import HealthCheckResponse, ReadinessCheckResponse
from src.inference.service import InferenceService
from src.storage.database import create_engine, init_db
from src.storage.repository import MarketDataRepository
from src.utils.config import load_config


class AppState:
    """Global application dependency state container."""

    def __init__(self):
        self.repository: Optional[MarketDataRepository] = None
        self.inference_service: Optional[InferenceService] = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown event handler.
    """
    # 1. Startup: Load configs &amp; initialize database connection
    config = load_config()
    db_url = config.get("database_url", "sqlite:///data/forecasting.db")
    engine = create_engine(db_url)
    init_db(engine)

    app_state.repository = MarketDataRepository(engine)
    app_state.inference_service = InferenceService(
        repository=app_state.repository,
        checkpoint_dir=config.get("paths", {}).get(
            "checkpoint_dir", "models/checkpoints"
        ),
        metadata_dir=config.get("paths", {}).get("metadata_dir", "models/metadata"),
    )
    yield
    # 2. Shutdown cleanup
    app_state.repository = None
    app_state.inference_service = None


app = FastAPI(
    title="Time Series Forecasting API",
    description="Production REST API exposing market data and ML price predictions",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(prediction.router)
app.include_router(market.router)
app.include_router(model.router)


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """
    Simple liveness check for container orchestrators.
    """
    return HealthCheckResponse(status="ok")


@app.get("/ready", response_model=ReadinessCheckResponse, tags=["Health"])
def readiness_check():
    """
    Readiness check verifying database repository and inference service state.
    """
    db_ready = app_state.repository is not None
    model_ready = app_state.inference_service is not None
    status = "ready" if (db_ready and model_ready) else "not_ready"
    return ReadinessCheckResponse(status=status, database=db_ready, model=model_ready)
