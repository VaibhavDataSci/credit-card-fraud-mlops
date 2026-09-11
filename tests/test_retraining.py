"""Lightweight retraining and candidate artifact tests."""

import pandas as pd
import pytest

from src.models.promotion import evaluate_predictions
from src.models.train import ModelTrainer


def tiny_processed_data(rows=40):
    return pd.DataFrame(
        {
            "type": (["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"] * (rows // 4 + 1))[:rows],
            "amount": [float(100 + index) for index in range(rows)],
            "oldbalanceOrg": [1000.0 + index for index in range(rows)],
            "newbalanceOrig": [900.0 + index for index in range(rows)],
            "oldbalanceDest": [50.0 + index for index in range(rows)],
            "newbalanceDest": [150.0 + index for index in range(rows)],
            "balance_diff_orig": [100.0] * rows,
            "balance_diff_dest": [100.0] * rows,
            "amount_to_balance_ratio": [0.1] * rows,
            "isFraud": ([0, 1] * (rows // 2)),
        }
    )


def test_target_is_separated_before_training():
    trainer = ModelTrainer({"model": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1}, "data_split": {"test_size": 0.25, "random_state": 42, "stratify": True}})
    X, y = trainer.load_data(tiny_processed_data())

    assert "isFraud" not in X.columns
    assert y.name == "isFraud"


def test_candidate_training_and_probability_sanity():
    trainer = ModelTrainer({"model": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1}, "data_split": {"test_size": 0.25, "random_state": 42, "stratify": True}})
    candidate, split = trainer.train(df=tiny_processed_data(), use_smote=False, use_scale_pos_weight=True)
    metrics = evaluate_predictions(candidate, split["X_test"], split["y_test"], threshold=0.90)

    assert metrics["test_samples"] == 10
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert candidate.named_steps["clf"].classes_.tolist() == [0, 1]
