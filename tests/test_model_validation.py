"""Unit tests for ModelValidator module."""

import os
import pytest
import mlflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.validate import ModelValidator

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@pytest.fixture
def dummy_run_environment(tmp_path):
    """Fixture providing a temporary MLflow run with logged metrics and params."""
    tracking_uri = f"file:{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_val_exp")

    model = LogisticRegression()
    X = pd.DataFrame({"feat1": [0.1, 0.2], "feat2": [0.5, 0.6]})
    y = pd.Series([0, 1])
    model.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "use_smote": "False",
            "use_scale_pos_weight": "True",
            "imbalance_strategy": "scale_pos_weight only, no SMOTE",
            "experiment_name": "scale_weight_only",
        })
        mlflow.log_metrics({
            "precision": 0.91,
            "recall": 0.99,
            "f1_score": 0.95,
            "roc_auc": 0.999,
            "pr_auc": 0.988,
        })
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        run_id = run.info.run_id

    return tracking_uri, run_id, model, X


def test_validator_pass(dummy_run_environment):
    """Test valid model candidate passes validation criteria."""
    tracking_uri, run_id, model, X = dummy_run_environment
    validator = ModelValidator(tracking_uri=tracking_uri)

    metadata = {
        "selected_experiment": "scale_weight_only",
        "selected_threshold": 0.90,
        "strategy": "scale_weight_only",
    }

    report = validator.validate_candidate(
        run_id=run_id,
        metadata=metadata,
        model=model,
        sample_input=X,
    )

    assert report["status"] == "PASSED"
    assert len(report["errors"]) == 0


def test_validator_invalid_strategy(dummy_run_environment):
    """Test candidate with invalid strategy fails validation."""
    tracking_uri, run_id, model, X = dummy_run_environment
    validator = ModelValidator(tracking_uri=tracking_uri)

    metadata = {
        "selected_experiment": "smote_only",
        "selected_threshold": 0.90,
        "strategy": "smote_only",
    }

    report = validator.validate_candidate(
        run_id=run_id,
        metadata=metadata,
        model=model,
        sample_input=X,
    )

    assert report["status"] == "FAILED"
    assert any("strategy" in e.lower() for e in report["errors"])


def test_validator_invalid_threshold(dummy_run_environment):
    """Test candidate with invalid threshold fails validation."""
    tracking_uri, run_id, model, X = dummy_run_environment
    validator = ModelValidator(tracking_uri=tracking_uri)

    metadata = {
        "selected_experiment": "scale_weight_only",
        "selected_threshold": 0.50,
        "strategy": "scale_weight_only",
    }

    report = validator.validate_candidate(
        run_id=run_id,
        metadata=metadata,
        model=model,
        sample_input=X,
    )

    assert report["status"] == "FAILED"
    assert any("threshold" in e.lower() for e in report["errors"])
