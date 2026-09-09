"""Unit tests for model evaluation module."""

import os
import numpy as np
import pandas as pd
import pytest
from src.models.evaluate import ModelEvaluator


class MockModel:
    """Mock model class providing predict and predict_proba."""

    def predict_proba(self, X):
        # Return mock probabilities
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 0.2
        probs[:, 1] = 0.8
        return probs

    def predict(self, X):
        return np.ones(len(X), dtype=int)


def test_model_evaluation_metrics_and_artifacts(tmp_path):
    """Test ModelEvaluator metrics calculation and artifact generation."""
    reports_dir = str(tmp_path / "reports")
    evaluator = ModelEvaluator(reports_dir=reports_dir)

    mock_pipeline = MockModel()
    X_test = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0]})
    y_test = pd.Series([0, 1, 0, 1])

    metrics = evaluator.evaluate(mock_pipeline, X_test, y_test)

    assert metrics["test_samples"] == 4
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

    # Check generated report files
    assert os.path.exists(os.path.join(reports_dir, "metrics.json"))
    assert os.path.exists(os.path.join(reports_dir, "classification_report.json"))
    assert os.path.exists(os.path.join(reports_dir, "confusion_matrix.png"))
    assert os.path.exists(os.path.join(reports_dir, "roc_curve.png"))
    assert os.path.exists(os.path.join(reports_dir, "precision_recall_curve.png"))
