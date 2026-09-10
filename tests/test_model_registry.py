"""Unit tests for MLflow Model Registry integration."""

import os
import pytest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.models.registry import ModelRegistry

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@pytest.fixture
def dummy_mlflow_run(tmp_path):
    """Fixture providing a temporary MLflow run with a dummy trained model."""
    tracking_uri = f"file:{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_registry_exp")

    model = LogisticRegression()
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    model.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        run_id = run.info.run_id

    return tracking_uri, run_id, model


def test_model_registry_flow(dummy_mlflow_run):
    """Test registering a model, attaching tags, assigning candidate/champion aliases, and loading."""
    tracking_uri, run_id, original_model = dummy_mlflow_run

    registry = ModelRegistry(tracking_uri=tracking_uri)
    model_name = "TestFraudDetector"

    tags = {
        "model_name": model_name,
        "model_strategy": "scale_weight_only",
        "threshold": "0.90",
        "validation_status": "passed",
    }

    # 1. Register model
    mv = registry.register_model(
        run_id=run_id,
        model_name=model_name,
        artifact_path="model",
        tags=tags,
    )

    assert mv is not None
    assert mv.name == model_name
    assert str(mv.version) == "1"

    # 2. Assign candidate and champion aliases
    registry.assign_alias(model_name, "candidate", mv.version)
    registry.assign_alias(model_name, "champion", mv.version)

    # Verify aliases on MLflow client
    client = MlflowClient(tracking_uri=tracking_uri)
    mv_cand = client.get_model_version_by_alias(model_name, "candidate")
    assert mv_cand.version == mv.version

    mv_champ = client.get_model_version_by_alias(model_name, "champion")
    assert mv_champ.version == mv.version

    # 3. Load registered model via champion alias
    loaded_model = registry.load_registered_model(model_name, "champion")
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")

    # Generate test prediction
    preds = loaded_model.predict([[0, 0], [1, 1]])
    assert len(preds) == 2
