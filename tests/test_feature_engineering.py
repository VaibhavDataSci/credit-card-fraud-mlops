"""Unit tests for feature engineering transformer."""

import pandas as pd
import pytest
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    """Fixture providing sample input transaction DataFrame."""
    return pd.DataFrame(
        {
            "amount": [100.0, 500.0],
            "oldbalanceOrg": [1000.0, 0.0],
            "newbalanceOrig": [900.0, 0.0],
            "oldbalanceDest": [0.0, 200.0],
            "newbalanceDest": [100.0, 700.0],
            "isFraud": [0, 1],
        }
    )


def test_balance_diff_orig_calculation(sample_df):
    """Test calculation of balance_diff_orig = oldbalanceOrg - newbalanceOrig."""
    engineer = FeatureEngineer({"enable_balance_diffs": True})
    transformed = engineer.transform(sample_df)

    assert "balance_diff_orig" in transformed.columns
    assert list(transformed["balance_diff_orig"]) == [100.0, 0.0]


def test_balance_diff_dest_calculation(sample_df):
    """Test calculation of balance_diff_dest = newbalanceDest - oldbalanceDest."""
    engineer = FeatureEngineer({"enable_balance_diffs": True})
    transformed = engineer.transform(sample_df)

    assert "balance_diff_dest" in transformed.columns
    assert list(transformed["balance_diff_dest"]) == [100.0, 500.0]


def test_amount_to_balance_ratio_calculation(sample_df):
    """Test calculation of amount_to_balance_ratio = amount / (oldbalanceOrg + 1.0)."""
    engineer = FeatureEngineer({"enable_amount_ratio": True})
    transformed = engineer.transform(sample_df)

    assert "amount_to_balance_ratio" in transformed.columns
    # Row 0: 100.0 / (1000.0 + 1.0) = 100 / 1001
    # Row 1: 500.0 / (0.0 + 1.0) = 500.0
    assert pytest.approx(transformed.loc[0, "amount_to_balance_ratio"], 0.001) == (100.0 / 1001.0)
    assert pytest.approx(transformed.loc[1, "amount_to_balance_ratio"], 0.001) == 500.0


def test_feature_engineering_disabled_config(sample_df):
    """Test that features are not created if disabled in configuration."""
    engineer = FeatureEngineer({"enable_balance_diffs": False, "enable_amount_ratio": False})
    transformed = engineer.transform(sample_df)

    assert "balance_diff_orig" not in transformed.columns
    assert "amount_to_balance_ratio" not in transformed.columns
