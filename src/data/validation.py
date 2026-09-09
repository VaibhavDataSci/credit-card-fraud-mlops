"""Data validation module for raw credit card fraud dataset."""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import pandas as pd


class DataValidator:
    """Automated validator for raw dataset quality, schema, and integrity."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize validator with configuration rules.

        Args:
            config: Validation configuration dictionary.
        """
        self.config = config
        self.raw_data_path = config.get("raw_data_path", "data/raw/AIML DATASET.csv")
        self.target_column = config.get("target_column", "isFraud")
        self.allowed_target_values = set(config.get("allowed_target_values", [0, 1]))
        self.required_columns = config.get("required_columns", [])
        self.numerical_columns = config.get("numerical_columns", [])
        self.categorical_columns = config.get("categorical_columns", [])

    def validate(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Run all validation checks and return a comprehensive report dictionary.

        Args:
            df: Optional DataFrame to validate directly (useful for testing).

        Returns:
            Structured validation report dictionary.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        checks: Dict[str, Any] = {}
        status_pass = True

        # 1. File Existence & Readability
        if df is None:
            if not os.path.exists(self.raw_data_path):
                return {
                    "timestamp": timestamp,
                    "dataset_path": self.raw_data_path,
                    "overall_status": "FAILED",
                    "error": f"Dataset file does not exist at {self.raw_data_path}",
                }
            try:
                df = pd.read_csv(self.raw_data_path)
            except Exception as e:
                return {
                    "timestamp": timestamp,
                    "dataset_path": self.raw_data_path,
                    "overall_status": "FAILED",
                    "error": f"Failed to read CSV: {str(e)}",
                }

        # 2. Dataset Size
        rows, cols = df.shape
        checks["dataset_size"] = {
            "rows": int(rows),
            "columns": int(cols),
            "status": "PASSED" if rows > 0 and cols > 0 else "FAILED",
        }
        if rows == 0 or cols == 0:
            status_pass = False

        # 3. Schema Check (Required & Unexpected Columns)
        existing_cols = set(df.columns)
        required_cols = set(self.required_columns)
        missing_cols = list(required_cols - existing_cols)
        unexpected_cols = list(existing_cols - required_cols)

        schema_status = "PASSED" if len(missing_cols) == 0 else "FAILED"
        if len(missing_cols) > 0:
            status_pass = False

        checks["schema"] = {
            "status": schema_status,
            "expected_columns": self.required_columns,
            "missing_columns": missing_cols,
            "unexpected_columns": unexpected_cols,
        }

        # 4. Data Types Check
        type_checks = {}
        type_status = "PASSED"

        for num_col in self.numerical_columns:
            if num_col in df.columns:
                is_num = pd.api.types.is_numeric_dtype(df[num_col])
                type_checks[num_col] = {
                    "expected": "numeric",
                    "actual": str(df[num_col].dtype),
                    "valid": bool(is_num),
                }
                if not is_num:
                    type_status = "FAILED"
                    status_pass = False

        for cat_col in self.categorical_columns:
            if cat_col in df.columns:
                is_cat = pd.api.types.is_object_dtype(df[cat_col]) or pd.api.types.is_string_dtype(
                    df[cat_col]
                )
                type_checks[cat_col] = {
                    "expected": "categorical/string",
                    "actual": str(df[cat_col].dtype),
                    "valid": bool(is_cat),
                }
                if not is_cat:
                    type_status = "FAILED"
                    status_pass = False

        checks["data_types"] = {
            "status": type_status,
            "details": type_checks,
        }

        # 5. Null Values Check
        null_counts = df.isnull().sum().to_dict()
        total_nulls = int(sum(null_counts.values()))
        null_percentages = {k: float((v / rows) * 100) for k, v in null_counts.items()}

        checks["null_values"] = {
            "status": "PASSED" if total_nulls == 0 else "WARNING",
            "total_nulls": total_nulls,
            "null_counts": {k: int(v) for k, v in null_counts.items()},
            "null_percentages": null_percentages,
        }

        # 6. Duplicate Records Check
        num_duplicates = int(df.duplicated().sum())
        dup_percentage = float((num_duplicates / rows) * 100) if rows > 0 else 0.0

        checks["duplicates"] = {
            "status": "PASSED" if num_duplicates == 0 else "WARNING",
            "duplicate_count": num_duplicates,
            "duplicate_percentage": dup_percentage,
        }

        # 7. Target Column Validation
        if self.target_column not in df.columns:
            checks["target_column"] = {
                "status": "FAILED",
                "error": f"Target column '{self.target_column}' missing.",
            }
            status_pass = False
        else:
            unique_targets = set(df[self.target_column].dropna().unique())
            invalid_targets = list(unique_targets - self.allowed_target_values)
            target_status = "PASSED" if len(invalid_targets) == 0 else "FAILED"
            if len(invalid_targets) > 0:
                status_pass = False

            val_counts = df[self.target_column].value_counts().to_dict()
            class_dist = {str(k): int(v) for k, v in val_counts.items()}
            fraud_count = int(val_counts.get(1, 0))
            non_fraud_count = int(val_counts.get(0, 0))
            fraud_pct = float((fraud_count / rows) * 100) if rows > 0 else 0.0

            checks["target_column"] = {
                "status": target_status,
                "target_name": self.target_column,
                "unique_values": [int(x) for x in unique_targets],
                "invalid_values": invalid_targets,
                "class_distribution": class_dist,
                "fraud_count": fraud_count,
                "non_fraud_count": non_fraud_count,
                "fraud_percentage": fraud_pct,
            }

        # 8. Numerical Sanity Checks (safely checked only for numeric series)
        sanity_issues = []

        if "amount" in df.columns and pd.api.types.is_numeric_dtype(df["amount"]):
            negative_amounts = int((df["amount"] < 0).sum())
            if negative_amounts > 0:
                sanity_issues.append(f"Found {negative_amounts} negative transaction amounts.")

        if "step" in df.columns and pd.api.types.is_numeric_dtype(df["step"]):
            invalid_steps = int((df["step"] < 1).sum())
            if invalid_steps > 0:
                sanity_issues.append(f"Found {invalid_steps} non-positive step values.")

        balance_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        for b_col in balance_cols:
            if b_col in df.columns and pd.api.types.is_numeric_dtype(df[b_col]):
                neg_b = int((df[b_col] < 0).sum())
                if neg_b > 0:
                    sanity_issues.append(f"Found {neg_b} negative balance values in '{b_col}'.")

        sanity_status = "PASSED" if len(sanity_issues) == 0 else "FAILED"
        if len(sanity_issues) > 0:
            status_pass = False

        checks["numerical_sanity"] = {
            "status": sanity_status,
            "issues": sanity_issues,
        }

        # Final Report Assembly
        report = {
            "timestamp": timestamp,
            "dataset_path": self.raw_data_path,
            "overall_status": "PASSED" if status_pass else "FAILED",
            "checks": checks,
        }

        return report
