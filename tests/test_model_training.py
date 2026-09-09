"""Unit tests for model training module."""

import os
import joblib
import pandas as pd
import pytest
from src.models.train import ModelTrainer


@pytest.fixture
def sample_processed_df():
    """Fixture providing sample processed transaction DataFrame."""
    return pd.DataFrame(
        {
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"] * 25,
            "amount": [100.0, 500.0, 200.0, 1000.0] * 25,
            "oldbalanceOrg": [1000.0, 500.0, 200.0, 1000.0] * 25,
            "newbalanceOrig": [900.0, 0.0, 0.0, 0.0] * 25,
            "oldbalanceDest": [0.0, 100.0, 0.0, 500.0] * 25,
            "newbalanceDest": [100.0, 600.0, 200.0, 1500.0] * 25,
            "balance_diff_orig": [100.0, 500.0, 200.0, 1000.0] * 25,
            "balance_diff_dest": [100.0, 500.0, 200.0, 1000.0] * 25,
            "amount_to_balance_ratio": [0.1, 1.0, 1.0, 1.0] * 25,
            "isFraud": [0, 0, 0, 1] * 25,
        }
    )


@pytest.fixture
def train_config(tmp_path):
    """Fixture providing model training configuration."""
    model_path = str(tmp_path / "test_model.joblib")
    return {
        "processed_data_path": "dummy.parquet",
        "target_column": "isFraud",
        "data_split": {"test_size": 0.3, "random_state": 42, "stratify": True},
        "smote": {"random_state": 42, "k_neighbors": 2},
        "model": {
            "name": "xgboost",
            "objective": "binary:logistic",
            "n_estimators": 5,
            "max_depth": 2,
            "random_state": 42,
            "n_jobs": 1,
        },
        "model_output_path": model_path,
    }


def test_model_trainer_split(sample_processed_df, train_config):
    """Test stratified train/test split proportions."""
    trainer = ModelTrainer(train_config)
    X, y = trainer.load_data(df=sample_processed_df)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    assert len(X_train) == 70
    assert len(X_test) == 30
    assert y_train.sum() == 17 or y_train.sum() == 18  # Stratified fraud count


def test_model_training_and_saving(sample_processed_df, train_config, tmp_path):
    """Test full training workflow and model artifact serialization."""
    out_file = str(tmp_path / "trained_model.joblib")
    train_config["model_output_path"] = out_file

    trainer = ModelTrainer(train_config)
    pipeline, metadata = trainer.train(df=sample_processed_df)
    saved_path = trainer.save_model(pipeline)

    assert os.path.exists(saved_path)
    loaded_pipeline = joblib.load(saved_path)
    assert hasattr(loaded_pipeline, "predict")

    # Predict on test set
    preds = loaded_pipeline.predict(metadata["X_test"])
    assert len(preds) == len(metadata["X_test"])
