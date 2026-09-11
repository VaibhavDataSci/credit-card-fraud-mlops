"""Tests for Prometheus exposition and low-cardinality instrumentation."""

from fastapi.testclient import TestClient
import re

from app.main import app, model_loader, monitoring
from app.prometheus_metrics import exposition


VALID_TRANSACTION = {
    "type": "TRANSFER",
    "amount": 1000.0,
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0,
}


def fake_loader(monkeypatch):
    def load():
        model_loader.loaded = True
        model_loader.version = "test-version"

    monkeypatch.setattr(model_loader, "load", load)
    monkeypatch.setattr(model_loader, "predict", lambda transaction: 0.97)
    monitoring.reset()


def test_metrics_endpoint_returns_prometheus_format(monkeypatch):
    fake_loader(monkeypatch)
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "# TYPE http_requests_total counter" in response.text
    assert "http_request_duration_seconds_bucket" in response.text


def test_request_prediction_and_quality_metrics_increment(monkeypatch):
    fake_loader(monkeypatch)
    before = exposition().decode()
    with TestClient(app) as client:
        client.post("/predict", json=VALID_TRANSACTION)
        client.post("/predict", json={"type": "UNKNOWN"})
        response = client.get("/metrics")

    assert response.status_code == 200
    def counter_value(body, name):
        match = re.search(rf"^{name} ([0-9.]+)$", body, re.MULTILINE)
        return float(match.group(1)) if match else 0.0

    assert counter_value(response.text, "fraud_predictions_total") > counter_value(before, "fraud_predictions_total")
    assert 'data_quality_errors_total{type="invalid_type"}' in response.text
    assert 'http_errors_total{endpoint="/predict",method="POST",status_class="4xx"}' in response.text
    assert "nameOrig" not in response.text
    assert "nameDest" not in response.text
    assert "TRANSFER" not in response.text
    assert response.text != before


def test_probability_is_histogram_not_label(monkeypatch):
    fake_loader(monkeypatch)
    with TestClient(app) as client:
        client.post("/predict", json=VALID_TRANSACTION)
        body = client.get("/metrics").text

    assert "fraud_probability_bucket" in body
    assert "fraud_probability=" not in body
    assert "0.97" not in body.split("fraud_probability_bucket", 1)[0]


def test_existing_monitoring_endpoint_remains_available(monkeypatch):
    fake_loader(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/monitoring")

    assert response.status_code == 200
    assert "requests" in response.json()
