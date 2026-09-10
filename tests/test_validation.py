"""Comprehensive tests for the project's data validation rules."""

import pandas as pd

from src.data.validation import DataValidator


REQUIRED_COLUMNS = [
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
]


VALIDATION_CONFIG = {
    "target_column": "isFraud",
    "allowed_target_values": [0, 1],
    "required_columns": REQUIRED_COLUMNS,
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


def valid_transactions() -> pd.DataFrame:
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


def test_valid_data_passes_validation():
    report = DataValidator(VALIDATION_CONFIG).validate(valid_transactions())

    assert report["overall_status"] == "PASSED"
    assert report["checks"]["schema"]["status"] == "PASSED"
    assert report["checks"]["target_column"]["status"] == "PASSED"


def test_missing_required_column_fails_validation():
    data = valid_transactions().drop(columns=["nameDest"])

    report = DataValidator(VALIDATION_CONFIG).validate(data)

    assert report["overall_status"] == "FAILED"
    assert "nameDest" in report["checks"]["schema"]["missing_columns"]


def test_invalid_target_values_fail_validation():
    data = valid_transactions()
    data.loc[0, "isFraud"] = 2

    report = DataValidator(VALIDATION_CONFIG).validate(data)

    assert report["overall_status"] == "FAILED"
    assert report["checks"]["target_column"]["status"] == "FAILED"
    assert 2 in report["checks"]["target_column"]["invalid_values"]


def test_invalid_data_type_fails_validation():
    data = valid_transactions()
    data["amount"] = ["one hundred", "five hundred"]

    report = DataValidator(VALIDATION_CONFIG).validate(data)

    assert report["overall_status"] == "FAILED"
    assert report["checks"]["data_types"]["status"] == "FAILED"


def test_null_values_are_reported_without_being_marked_as_validation_failure():
    data = valid_transactions()
    data.loc[0, "amount"] = None

    report = DataValidator(VALIDATION_CONFIG).validate(data)

    assert report["checks"]["null_values"]["total_nulls"] == 1
    assert report["checks"]["null_values"]["status"] == "WARNING"


def test_negative_amount_and_balance_fail_numeric_sanity():
    data = valid_transactions()
    data.loc[0, "amount"] = -1
    data.loc[1, "oldbalanceOrg"] = -0.01

    report = DataValidator(VALIDATION_CONFIG).validate(data)

    assert report["overall_status"] == "FAILED"
    assert report["checks"]["numerical_sanity"]["status"] == "FAILED"
    assert len(report["checks"]["numerical_sanity"]["issues"]) == 2


def test_non_positive_step_fails_numeric_sanity():
    data = valid_transactions()
    data.loc[0, "step"] = 0

    report = DataValidator(VALIDATION_CONFIG).validate(data)

    assert report["overall_status"] == "FAILED"
    assert any("step" in issue for issue in report["checks"]["numerical_sanity"]["issues"])
