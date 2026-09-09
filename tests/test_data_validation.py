"""Unit tests for data validation logic."""

import pytest
import pandas as pd
from src.data.validation import DataValidator


@pytest.fixture
def base_config():
    """Fixture providing baseline validation configuration."""
    return {
        "raw_data_path": "dummy.csv",
        "target_column": "isFraud",
        "allowed_target_values": [0, 1],
        "required_columns": [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        ],
        "numerical_columns": [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        ],
        "categorical_columns": ["type", "nameOrig", "nameDest"],
    }


@pytest.fixture
def valid_df():
    """Fixture providing a small valid mock DataFrame."""
    return pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["PAYMENT", "TRANSFER"],
            "amount": [100.0, 500.0],
            "nameOrig": ["C123", "C456"],
            "oldbalanceOrg": [1000.0, 500.0],
            "newbalanceOrig": [900.0, 0.0],
            "nameDest": ["M789", "C999"],
            "oldbalanceDest": [0.0, 100.0],
            "newbalanceDest": [100.0, 600.0],
            "isFraud": [0, 1],
            "isFlaggedFraud": [0, 0],
        }
    )


def test_valid_dataframe_passes(base_config, valid_df):
    """Test that a completely valid DataFrame passes all checks."""
    validator = DataValidator(base_config)
    report = validator.validate(df=valid_df)

    assert report["overall_status"] == "PASSED"
    assert report["checks"]["schema"]["status"] == "PASSED"
    assert report["checks"]["target_column"]["status"] == "PASSED"
    assert report["checks"]["target_column"]["fraud_count"] == 1
    assert report["checks"]["target_column"]["non_fraud_count"] == 1


def test_missing_required_column_fails(base_config, valid_df):
    """Test that missing a required column causes validation to fail."""
    invalid_df = valid_df.drop(columns=["isFraud"])
    validator = DataValidator(base_config)
    report = validator.validate(df=invalid_df)

    assert report["overall_status"] == "FAILED"
    assert report["checks"]["schema"]["status"] == "FAILED"
    assert "isFraud" in report["checks"]["schema"]["missing_columns"]


def test_invalid_target_values_fail(base_config, valid_df):
    """Test that target values outside allowed set {0, 1} trigger failure."""
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "isFraud"] = 99  # Invalid class
    validator = DataValidator(base_config)
    report = validator.validate(df=invalid_df)

    assert report["overall_status"] == "FAILED"
    assert report["checks"]["target_column"]["status"] == "FAILED"
    assert 99 in report["checks"]["target_column"]["invalid_values"]


def test_negative_amount_fails_sanity(base_config, valid_df):
    """Test that negative transaction amounts fail numerical sanity check."""
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "amount"] = -50.0
    validator = DataValidator(base_config)
    report = validator.validate(df=invalid_df)

    assert report["overall_status"] == "FAILED"
    assert report["checks"]["numerical_sanity"]["status"] == "FAILED"
    assert len(report["checks"]["numerical_sanity"]["issues"]) > 0


def test_null_values_warning(base_config, valid_df):
    """Test that null values are correctly detected and reported."""
    null_df = valid_df.copy()
    null_df.loc[0, "amount"] = None
    validator = DataValidator(base_config)
    report = validator.validate(df=null_df)

    assert report["checks"]["null_values"]["total_nulls"] == 1
    assert report["checks"]["null_values"]["null_counts"]["amount"] == 1


def test_duplicate_rows_detection(base_config, valid_df):
    """Test duplicate rows detection."""
    dup_df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
    validator = DataValidator(base_config)
    report = validator.validate(df=dup_df)

    assert report["checks"]["duplicates"]["duplicate_count"] == 1


def test_data_type_mismatch_fails(base_config, valid_df):
    """Test that wrong data type in numerical column fails validation."""
    type_df = valid_df.copy()
    type_df["amount"] = ["hundred", "five_hundred"]  # String instead of numeric
    validator = DataValidator(base_config)
    report = validator.validate(df=type_df)

    assert report["overall_status"] == "FAILED"
    assert report["checks"]["data_types"]["status"] == "FAILED"
