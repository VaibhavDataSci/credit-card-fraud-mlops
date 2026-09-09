"""Data preprocessing module for credit card fraud detection pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd

from src.features.feature_engineering import FeatureEngineer


class DataPreprocessor:
    """Production preprocessor for raw transaction dataset."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration settings.

        Args:
            config: Preprocessing configuration dictionary.
        """
        self.config = config
        self.raw_data_path = config.get("raw_data_path", "data/raw/AIML DATASET.csv")
        self.processed_data_path = config.get("processed_data_path", "data/processed/cleaned.parquet")
        self.target_column = config.get("target_column", "isFraud")
        self.drop_columns = config.get("drop_columns", ["nameOrig", "nameDest", "isFlaggedFraud", "step"])
        self.categorical_columns = config.get("categorical_columns", ["type"])

    def process(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run preprocessing pipeline and feature engineering.

        Args:
            df: Optional DataFrame to process directly.

        Returns:
            Tuple of (processed DataFrame, processing report details dictionary).
        """
        if df is None:
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"Raw dataset file not found at {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)

        input_shape = df.shape
        input_columns = list(df.columns)

        # 1. Feature Engineering
        fe_config = self.config.get("feature_engineering", {})
        engineer = FeatureEngineer(fe_config)
        processed_df = engineer.transform(df)

        # Identify created features
        created_features = [c for c in processed_df.columns if c not in input_columns]

        # 2. Drop high-cardinality identifiers and constant/irrelevant columns
        cols_to_drop = [c for c in self.drop_columns if c in processed_df.columns]
        processed_df = processed_df.drop(columns=cols_to_drop)

        # 3. Ensure target is present
        if self.target_column not in processed_df.columns:
            raise KeyError(f"Target column '{self.target_column}' missing after preprocessing!")

        # Ensure categorical column dtypes
        for cat_col in self.categorical_columns:
            if cat_col in processed_df.columns:
                processed_df[cat_col] = processed_df[cat_col].astype("category")

        output_shape = processed_df.shape
        output_columns = list(processed_df.columns)

        metadata = {
            "input_shape": {"rows": int(input_shape[0]), "columns": int(input_shape[1])},
            "output_shape": {"rows": int(output_shape[0]), "columns": int(output_shape[1])},
            "features_removed": cols_to_drop,
            "features_created": created_features,
            "final_columns": output_columns,
            "target_column": self.target_column,
        }

        return processed_df, metadata

    def save_processed_data(self, df: pd.DataFrame, output_path: str = None) -> str:
        """Save processed DataFrame to Parquet format.

        Args:
            df: Processed DataFrame.
            output_path: Path to save Parquet file.

        Returns:
            Resolved output path string.
        """
        save_path = output_path or self.processed_data_path
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df.to_parquet(save_path, index=False)
        return save_path
