"""Unit tests for data preprocessing module."""

import os
import pandas as pd
import pytest
from src.data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_raw_df():
    """Fixture providing raw transaction sample DataFrame."""
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


@pytest.fixture
def prep_config():
    """Fixture providing preprocessing configuration."""
    return {
        "raw_data_path": "dummy.csv",
        "processed_data_path": "data/processed/test_cleaned.parquet",
        "target_column": "isFraud",
        "drop_columns": ["nameOrig", "nameDest", "isFlaggedFraud", "step"],
        "categorical_columns": ["type"],
        "feature_engineering": {"enable_balance_diffs": True, "enable_amount_ratio": True},
    }


def test_preprocessing_pipeline_execution(sample_raw_df, prep_config):
    """Test end-to-end preprocessing execution on sample DataFrame."""
    preprocessor = DataPreprocessor(prep_config)
    processed_df, metadata = preprocessor.process(df=sample_raw_df)

    # 1. Verify high-cardinality identifiers dropped
    assert "nameOrig" not in processed_df.columns
    assert "nameDest" not in processed_df.columns
    assert "isFlaggedFraud" not in processed_df.columns
    assert "step" not in processed_df.columns

    # 2. Verify engineered features present
    assert "balance_diff_orig" in processed_df.columns
    assert "balance_diff_dest" in processed_df.columns
    assert "amount_to_balance_ratio" in processed_df.columns

    # 3. Verify target preserved
    assert "isFraud" in processed_df.columns
    assert list(processed_df["isFraud"]) == [0, 1]

    # 4. Verify category dtype
    assert str(processed_df["type"].dtype) == "category"

    # 5. Verify metadata details
    assert metadata["input_shape"] == {"rows": 2, "columns": 11}
    assert metadata["output_shape"] == {"rows": 2, "columns": 10}


def test_missing_target_raises_error(sample_raw_df, prep_config):
    """Test that missing target column raises KeyError."""
    invalid_df = sample_raw_df.drop(columns=["isFraud"])
    preprocessor = DataPreprocessor(prep_config)

    with pytest.raises(KeyError, match="Target column 'isFraud' missing"):
        preprocessor.process(df=invalid_df)


def test_save_processed_data_parquet(sample_raw_df, prep_config, tmp_path):
    """Test saving processed DataFrame to Parquet format."""
    out_file = str(tmp_path / "test_out.parquet")
    preprocessor = DataPreprocessor(prep_config)
    processed_df, _ = preprocessor.process(df=sample_raw_df)

    saved_path = preprocessor.save_processed_data(processed_df, output_path=out_file)

    assert os.path.exists(saved_path)
    loaded_df = pd.read_parquet(saved_path)
    assert loaded_df.shape == processed_df.shape
    assert "balance_diff_orig" in loaded_df.columns
