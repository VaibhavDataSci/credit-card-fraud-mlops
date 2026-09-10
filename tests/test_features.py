"""Comprehensive tests for the production feature engineering transformer."""

import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


def transaction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": ["TRANSFER", "PAYMENT"],
            "amount": [100.0, 0.0],
            "oldbalanceOrg": [1000.0, 0.0],
            "newbalanceOrig": [900.0, 0.0],
            "oldbalanceDest": [50.0, 0.0],
            "newbalanceDest": [150.0, 0.0],
            "isFraud": [0, 1],
        }
    )


def test_feature_creation_preserves_original_columns():
    source = transaction_frame()
    transformed = FeatureEngineer().transform(source)

    assert set(source.columns).issubset(transformed.columns)
    assert {
        "balance_diff_orig",
        "balance_diff_dest",
        "amount_to_balance_ratio",
    }.issubset(transformed.columns)


def test_engineered_feature_values_are_correct():
    transformed = FeatureEngineer().transform(transaction_frame())

    assert transformed["balance_diff_orig"].tolist() == [100.0, 0.0]
    assert transformed["balance_diff_dest"].tolist() == [100.0, 0.0]
    assert transformed.loc[0, "amount_to_balance_ratio"] == pytest.approx(100 / 1001)
    assert transformed.loc[1, "amount_to_balance_ratio"] == pytest.approx(0.0)


def test_zero_balances_and_zero_amount_do_not_divide_by_zero():
    source = pd.DataFrame(
        {
            "amount": [0.0],
            "oldbalanceOrg": [0.0],
            "newbalanceOrig": [0.0],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [0.0],
        }
    )

    transformed = FeatureEngineer().transform(source)

    assert transformed.loc[0, "amount_to_balance_ratio"] == 0.0
    assert transformed.loc[0, "balance_diff_orig"] == 0.0
    assert transformed.loc[0, "balance_diff_dest"] == 0.0


def test_feature_engineering_is_deterministic():
    engineer = FeatureEngineer()
    first = engineer.transform(transaction_frame())
    second = engineer.transform(transaction_frame())

    pd.testing.assert_frame_equal(first, second)


def test_disabled_features_are_not_created():
    transformed = FeatureEngineer(
        {"enable_balance_diffs": False, "enable_amount_ratio": False}
    ).transform(transaction_frame())

    assert "balance_diff_orig" not in transformed
    assert "balance_diff_dest" not in transformed
    assert "amount_to_balance_ratio" not in transformed
