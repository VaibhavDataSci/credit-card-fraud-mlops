"""Unit tests for MLflow experiment tracking module."""

import os
import mlflow
import pytest
from sklearn.linear_model import LogisticRegression
from src.models.tracking import MLflowTracker


@pytest.fixture
def mlflow_config(tmp_path):
    """Fixture providing MLflow configuration with isolated temporary tracking URI."""
    tracking_dir = str(tmp_path / "mlruns")
    return {
        "tracking_uri": f"file:{tracking_dir}",
        "experiment_name": "test_experiment",
        "run_name": "test_run",
    }


def test_mlflow_tracker_lifecycle(mlflow_config, tmp_path):
    """Test MLflowTracker run start, logging params, metrics, artifacts, models, and ending run."""
    tracker = MLflowTracker(mlflow_config)

    # 1. Start run
    run = tracker.start_run()
    assert run is not None
    assert mlflow.active_run() is not None

    # 2. Log parameters
    params = {"data_split_test_size": 0.3, "model_type": "xgboost"}
    tracker.log_params(params)

    # 3. Log metrics
    metrics = {"precision": 0.95, "recall": 0.98, "f1_score": 0.96}
    tracker.log_metrics(metrics)

    # 4. Log dummy artifact
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "test.txt"), "w") as f:
        f.write("test content")
    tracker.log_artifacts(reports_dir)

    # 5. Log real sklearn model (LogisticRegression)
    model = LogisticRegression()
    model.fit([[0], [1]], [0, 1])
    tracker.log_model(model, artifact_path="test_model")

    # 6. End run
    tracker.end_run()
    assert mlflow.active_run() is None


def test_mlflow_experiment_creation(mlflow_config):
    """Test that MLflow experiment is set up correctly."""
    tracker = MLflowTracker(mlflow_config)
    exp = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])

    assert exp is not None
    assert exp.name == "test_experiment"
