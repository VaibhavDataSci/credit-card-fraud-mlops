"""Evidently-based feature drift detection for production-like data."""

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

FEATURE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
    "amount_to_balance_ratio",
]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column != "type"]
CATEGORICAL_FEATURES = ["type"]


class DriftDataError(ValueError):
    """Raised when reference/current data cannot be compared safely."""


class DriftDetector:
    """Compare compatible reference and current feature distributions."""

    def __init__(self, drift_share_threshold: float = 0.5) -> None:
        if not 0 < drift_share_threshold <= 1:
            raise ValueError("drift_share_threshold must be between 0 and 1")
        self.drift_share_threshold = drift_share_threshold

    @staticmethod
    def load_data(path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Drift data not found: {path}")
        if path.suffix.lower() in {".parquet", ".pq"} and sample_size:
            try:
                import pyarrow.parquet as parquet

                parquet_file = parquet.ParquetFile(path)
                available_columns = set(parquet_file.schema.names)
                columns = [
                    column for column in FEATURE_COLUMNS + ["isFraud"] if column in available_columns
                ]
                batches = parquet_file.iter_batches(
                    batch_size=sample_size, columns=columns
                )
                data = next(batches, None)
                if data is None:
                    raise DriftDataError(f"Drift data is empty: {path}")
                return data.to_pandas()
            except ImportError:
                pass
        if path.suffix.lower() in {".parquet", ".pq"}:
            data = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            data = pd.read_csv(path)
        else:
            raise DriftDataError(f"Unsupported drift data format: {path.suffix}")
        if sample_size:
            data = data.head(sample_size)
        return data

    @staticmethod
    def validate_data(data: pd.DataFrame, name: str = "data") -> None:
        if data.empty:
            raise DriftDataError(f"{name} dataset is empty")
        missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
        if missing:
            raise DriftDataError(f"{name} dataset missing required features: {missing}")
        for column in NUMERIC_FEATURES:
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise DriftDataError(f"{name} feature '{column}' must be numeric")
            values = data[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise DriftDataError(f"{name} feature '{column}' contains non-finite values")
        if data["type"].isna().any():
            raise DriftDataError(f"{name} feature 'type' contains missing values")

    def compare(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
        self.validate_data(reference, "reference")
        self.validate_data(current, "current")
        reference_features = reference[FEATURE_COLUMNS].copy()
        current_features = current[FEATURE_COLUMNS].copy()
        reference_features["type"] = reference_features["type"].astype(str)
        current_features["type"] = current_features["type"].astype(str)

        snapshot = Report(
            [
                DataDriftPreset(
                    columns=FEATURE_COLUMNS,
                    drift_share=self.drift_share_threshold,
                    include_tests=True,
                )
            ]
        ).run(current_features, reference_features)
        raw = snapshot.dict()
        feature_results: dict[str, dict[str, Any]] = {}
        drifted_features: list[str] = []
        for metric in raw.get("metrics", []):
            name = metric.get("metric_name", "")
            if not name.startswith("ValueDrift("):
                continue
            feature = metric["config"]["column"]
            value = metric.get("value")
            threshold = metric.get("config", {}).get("threshold", 0.1)
            method = str(metric.get("config", {}).get("method", ""))
            if "p_value" in method:
                drifted = isinstance(value, (int, float, np.number)) and value < threshold
            else:
                drifted = isinstance(value, (int, float, np.number)) and value >= threshold
            feature_results[feature] = {
                "drift_score": value,
                "drift_threshold": threshold,
                "drift_detected": drifted,
            }
            if drifted:
                drifted_features.append(feature)

        drift_share = len(drifted_features) / len(FEATURE_COLUMNS)
        return {
            "reference_rows": int(len(reference_features)),
            "current_rows": int(len(current_features)),
            "features_analyzed": FEATURE_COLUMNS,
            "drifted_features": drifted_features,
            "drift_share": drift_share,
            "drift_share_threshold": self.drift_share_threshold,
            "dataset_drift_detected": drift_share >= self.drift_share_threshold,
            "investigation_required": bool(drifted_features),
            "feature_results": feature_results,
            "missing_values": {
                "reference": {key: int(value) for key, value in reference_features.isna().sum().items()},
                "current": {key: int(value) for key, value in current_features.isna().sum().items()},
            },
            "interpretation": "Distributional drift is an investigation signal, not automatic model failure or replacement.",
            "evidently_report": snapshot,
        }

    def save_reports(self, result: dict[str, Any], html_path: str | Path, json_path: str | Path) -> None:
        html_path = Path(html_path)
        json_path = Path(json_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = result.pop("evidently_report")
        snapshot.save_html(str(html_path))
        json_path.write_text(
            json.dumps(result, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
            encoding="utf-8",
        )


def run_drift_detection(
    reference_path: str | Path,
    current_path: str | Path,
    html_path: str | Path,
    json_path: str | Path,
    sample_size: int = 10000,
    drift_share_threshold: float = 0.5,
) -> dict[str, Any]:
    detector = DriftDetector(drift_share_threshold)
    reference = detector.load_data(reference_path, sample_size=sample_size)
    current = detector.load_data(current_path, sample_size=sample_size)
    result = detector.compare(reference, current)
    detector.save_reports(result, html_path, json_path)
    return result
