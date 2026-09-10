"""Tests for the deployed Champion model and its loader."""

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from app.model_loader import ChampionModelLoader
from src.features.feature_engineering import FeatureEngineer

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


TRANSACTION = {
    "type": "TRANSFER",
    "amount": 1000.0,
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0,
}


@pytest.fixture(scope="module")
def champion_loader():
    loader = ChampionModelLoader()
    try:
        loader.load()
    except Exception as exc:
        pytest.skip(f"local Champion artifact unavailable: {exc}")
    return loader


def test_loader_resolves_champion_uri_without_registry_mutation(monkeypatch):
    loader = ChampionModelLoader()
    calls = []

    class FakeClient:
        def __init__(self, tracking_uri):
            calls.append(("client", tracking_uri))

        def get_model_version_by_alias(self, name, alias):
            calls.append(("alias", name, alias))
            return SimpleNamespace(
                version="test-version",
                run_id="test-run",
                tags={"model_strategy": "scale_weight_only", "threshold": "0.90"},
            )

    fake_model = SimpleNamespace(classes_=[0, 1])
    monkeypatch.setattr("app.model_loader.MlflowClient", FakeClient)
    monkeypatch.setattr("app.model_loader.mlflow.set_tracking_uri", lambda uri: calls.append(("tracking", uri)))
    monkeypatch.setattr(
        "app.model_loader.mlflow.sklearn",
        SimpleNamespace(load_model=lambda uri: calls.append(("load", uri)) or fake_model),
    )

    loader.load()

    assert ("alias", "CreditCardFraudDetector", "champion") in calls
    assert ("load", "models:/CreditCardFraudDetector@champion") in calls
    assert loader.info()["threshold"] == pytest.approx(0.90)


def test_champion_model_loads(champion_loader):
    assert champion_loader.loaded is True
    assert champion_loader.model_name == "CreditCardFraudDetector"
    assert champion_loader.alias == "champion"
    assert champion_loader.model_uri == "models:/CreditCardFraudDetector@champion"
    assert champion_loader.version is not None
    assert champion_loader.run_id is not None


def test_champion_metadata_matches_phase_configuration(champion_loader):
    metadata = champion_loader.info()

    assert metadata["strategy"] == "scale_weight_only"
    assert metadata["threshold"] == pytest.approx(0.90)
    assert metadata["model_name"] == "CreditCardFraudDetector"
    assert metadata["alias"] == "champion"


def test_prediction_probability_is_valid_and_deterministic(champion_loader):
    first = champion_loader.predict(TRANSACTION)
    second = champion_loader.predict(TRANSACTION)

    assert 0.0 <= first <= 1.0
    assert first == pytest.approx(second)


def test_fraud_probability_uses_class_one(champion_loader):
    raw = pd.DataFrame([TRANSACTION])
    features = FeatureEngineer().transform(raw)
    probabilities = champion_loader.model.predict_proba(features)[0]
    classes = champion_loader.model.classes_
    fraud_index = next(index for index, value in enumerate(classes) if int(value) == 1)
    assert classes.tolist() == [0, 1]
    assert champion_loader.predict(TRANSACTION) == pytest.approx(probabilities[fraud_index])


def test_threshold_uses_ninety_percent(champion_loader):
    probability = champion_loader.predict(TRANSACTION)

    assert champion_loader.threshold == pytest.approx(0.90)
    assert (probability >= champion_loader.threshold) is True
    assert (0.899999 >= champion_loader.threshold) is False
    assert (0.90 >= champion_loader.threshold) is True
