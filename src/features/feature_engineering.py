"""Feature engineering module for financial fraud detection dataset."""

from typing import Any, Dict
import pandas as pd


class FeatureEngineer:
    """Vectorized feature engineering transformer."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize feature engineer with optional configuration settings.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.enable_balance_diffs = self.config.get("enable_balance_diffs", True)
        self.enable_amount_ratio = self.config.get("enable_amount_ratio", True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features on a copy of the input DataFrame.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Transformed DataFrame containing new engineered features.
        """
        data = df.copy()

        # 1. Balance differences
        if self.enable_balance_diffs:
            if "oldbalanceOrg" in data.columns and "newbalanceOrig" in data.columns:
                data["balance_diff_orig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]

            if "newbalanceDest" in data.columns and "oldbalanceDest" in data.columns:
                data["balance_diff_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]

        # 2. Transaction Amount to Origin Balance ratio
        if self.enable_amount_ratio:
            if "amount" in data.columns and "oldbalanceOrg" in data.columns:
                data["amount_to_balance_ratio"] = data["amount"] / (data["oldbalanceOrg"] + 1.0)

        return data
