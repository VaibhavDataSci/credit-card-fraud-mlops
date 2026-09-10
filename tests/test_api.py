"""Tests for the Phase 9 FastAPI inference service."""

from fastapi.testclient import TestClient

from app.main import app, model_loader


VALID_TRANSACTION = {
    "type": "TRANSFER",
    "amount": 1000.0,
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0,
}


def configure_fake_loader(monkeypatch, loaded=True):
    """Configure the application singleton without touching the registry."""
    def fake_load():
        if not loaded:
            model_loader.loaded = False
            raise RuntimeError("controlled model loading failure")
        model_loader.loaded = True
        model_loader.version = "test-version"
        model_loader.run_id = "test-run"

    monkeypatch.setattr(model_loader, "load", fake_load)
    monkeypatch.setattr(model_loader, "predict", lambda transaction: 0.97)


def test_health_when_model_loads(monkeypatch):
    configure_fake_loader(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_info_exposes_champion_configuration(monkeypatch):
    configure_fake_loader(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/model-info")

    body = response.json()
    assert response.status_code == 200
    assert body["model_name"] == "CreditCardFraudDetector"
    assert body["alias"] == "champion"
    assert body["version"] == "test-version"
    assert body["run_id"] == "test-run"
    assert body["strategy"] == "scale_weight_only"
    assert body["threshold"] == 0.90


def test_predict_returns_probability_and_threshold(monkeypatch):
    configure_fake_loader(monkeypatch)
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_TRANSACTION)

    body = response.json()
    assert response.status_code == 200
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["is_fraud"] is True
    assert body["threshold"] == 0.90
    assert body["model_name"] == "CreditCardFraudDetector"
    assert body["model_alias"] == "champion"


def test_predict_rejects_invalid_input(monkeypatch):
    configure_fake_loader(monkeypatch)
    invalid_transaction = {**VALID_TRANSACTION, "amount": -1}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_transaction)

    assert response.status_code == 422


def test_predict_rejects_missing_required_field(monkeypatch):
    configure_fake_loader(monkeypatch)
    invalid_transaction = {key: value for key, value in VALID_TRANSACTION.items() if key != "amount"}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_transaction)

    assert response.status_code == 422


def test_predict_rejects_invalid_transaction_type(monkeypatch):
    configure_fake_loader(monkeypatch)
    invalid_transaction = {**VALID_TRANSACTION, "type": "UNKNOWN"}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_transaction)

    assert response.status_code == 422


def test_predict_does_not_accept_target_as_model_feature(monkeypatch):
    configure_fake_loader(monkeypatch)
    received = {}

    def fake_predict(transaction):
        received.update(transaction)
        return 0.10

    monkeypatch.setattr(model_loader, "predict", fake_predict)
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID_TRANSACTION, "isFraud": 1})

    assert response.status_code == 200
    assert "isFraud" not in received
    assert response.json()["is_fraud"] is False


def test_predict_returns_503_when_model_is_unavailable(monkeypatch):
    configure_fake_loader(monkeypatch, loaded=False)
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_TRANSACTION)

    assert response.status_code == 503
    assert response.json()["detail"] == "Champion model is unavailable"


def test_model_loader_runs_once_per_application_startup(monkeypatch):
    calls = []

    def fake_load():
        calls.append("load")
        model_loader.loaded = True

    monkeypatch.setattr(model_loader, "load", fake_load)
    monkeypatch.setattr(model_loader, "predict", lambda transaction: 0.10)
    with TestClient(app) as client:
        client.get("/health")
        client.post("/predict", json=VALID_TRANSACTION)
        client.post("/predict", json=VALID_TRANSACTION)

    assert calls == ["load"]
