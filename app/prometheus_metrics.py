"""Prometheus metrics for the FastAPI fraud service."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


registry = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the API.",
    ["method", "endpoint", "status"],
    registry=registry,
)
HTTP_ERRORS = Counter(
    "http_errors_total",
    "Total HTTP 4xx and 5xx responses.",
    ["method", "endpoint", "status_class"],
    registry=registry,
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "endpoint", "status"],
    registry=registry,
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Predictions classified as fraud.",
    registry=registry,
)
NON_FRAUD_PREDICTIONS = Counter(
    "non_fraud_predictions_total",
    "Predictions classified as non-fraud.",
    registry=registry,
)
FRAUD_PROBABILITY = Histogram(
    "fraud_probability",
    "Distribution of class-1 fraud probabilities.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry,
)
DATA_QUALITY_ERRORS = Counter(
    "data_quality_errors_total",
    "Validation/data-quality errors by controlled category.",
    ["type"],
    registry=registry,
)
MODEL_INFO = Gauge(
    "model_info",
    "Loaded model metadata; value is always 1 when reported.",
    ["model_name", "model_alias", "model_version"],
    registry=registry,
)


def record_request(method: str, endpoint: str, status_code: int, duration_seconds: float) -> None:
    status = str(status_code)
    HTTP_REQUESTS.labels(method, endpoint, status).inc()
    HTTP_DURATION.labels(method, endpoint, status).observe(duration_seconds)
    if status_code >= 400:
        HTTP_ERRORS.labels(method, endpoint, f"{status_code // 100}xx").inc()


def record_prediction(is_fraud: bool, probability: float) -> None:
    (FRAUD_PREDICTIONS if is_fraud else NON_FRAUD_PREDICTIONS).inc()
    FRAUD_PROBABILITY.observe(probability)


def record_data_quality(category: str) -> None:
    allowed = {"missing_field", "invalid_type", "invalid_amount", "invalid_balance", "invalid_numeric_value"}
    DATA_QUALITY_ERRORS.labels(category if category in allowed else "invalid_type").inc()


def set_model_info(model_name: str, alias: str, version: str | None) -> None:
    if version:
        MODEL_INFO.labels(model_name, alias, version).set(1)


def exposition() -> bytes:
    return generate_latest(registry)
