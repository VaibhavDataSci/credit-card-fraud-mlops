"""FastAPI service for CreditCardFraudDetector Champion inference."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from app.model_loader import ChampionModelLoader
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model_loader = ChampionModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the registry model once during application startup."""
    try:
        model_loader.load()
    except Exception:
        logger.error("Fraud API started without an available Champion model")
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Inference service backed by the MLflow CreditCardFraudDetector Champion model.",
    version="9.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy" if model_loader.loaded else "unavailable", model_loaded=model_loader.loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(transaction: TransactionRequest, request: Request) -> PredictionResponse:
    if not model_loader.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Champion model is unavailable")
    try:
        probability = model_loader.predict(transaction.model_dump())
        return PredictionResponse(
            is_fraud=probability >= model_loader.threshold,
            fraud_probability=probability,
            threshold=model_loader.threshold,
            model_name=model_loader.model_name,
            model_alias=model_loader.alias,
        )
    except Exception:
        logger.exception("Prediction failed for request %s", request.url.path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed") from None


@app.get("/model-info", response_model=ModelInfoResponse, tags=["service"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_loader.info())
