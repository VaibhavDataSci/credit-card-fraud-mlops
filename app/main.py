"""FastAPI service for CreditCardFraudDetector Champion inference."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

from app.model_loader import ChampionModelLoader
from app.monitoring import MonitoringState
from app import prometheus_metrics
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model_loader = ChampionModelLoader()
monitoring = MonitoringState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the registry model once during application startup."""
    try:
        model_loader.load()
        prometheus_metrics.set_model_info(model_loader.model_name, model_loader.alias, model_loader.version)
    except Exception:
        logger.error("Fraud API started without an available Champion model")
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Inference service backed by the MLflow CreditCardFraudDetector Champion model.",
    version="9.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    started_at = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = (time.perf_counter() - started_at) * 1000
        status_code = response.status_code if response is not None else 500
        monitoring.record_request(request.url.path, request.method, status_code, latency_ms)
        prometheus_metrics.record_request(request.method, request.url.path, status_code, latency_ms / 1000)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    for error in exc.errors():
        location = error.get("loc", ())
        field = location[-1] if location else "request"
        error_type = str(error.get("type", ""))
        if error_type == "missing":
            category = "missing_field"
        elif field in {"amount"}:
            category = "invalid_amount"
        elif field in {"oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}:
            category = "invalid_balance"
        else:
            category = "invalid_type"
        monitoring.record_data_quality_error(category)
        prometheus_metrics.record_data_quality(category)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy" if model_loader.loaded else "unavailable", model_loaded=model_loader.loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(transaction: TransactionRequest, request: Request) -> PredictionResponse:
    if not model_loader.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Champion model is unavailable")
    try:
        probability = model_loader.predict(transaction.model_dump())
        prediction = PredictionResponse(
            is_fraud=probability >= model_loader.threshold,
            fraud_probability=probability,
            threshold=model_loader.threshold,
            model_name=model_loader.model_name,
            model_alias=model_loader.alias,
        )
        monitoring.record_prediction(probability, prediction.is_fraud, model_loader.threshold)
        prometheus_metrics.record_prediction(prediction.is_fraud, probability)
        return prediction
    except Exception:
        logger.exception("Prediction failed for request %s", request.url.path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed") from None


@app.get("/model-info", response_model=ModelInfoResponse, tags=["service"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_loader.info())


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=prometheus_metrics.exposition(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring", tags=["monitoring"])
def monitoring_summary() -> dict:
    return monitoring.summary()
