"""MLflow experiment tracking manager module."""

import os
from typing import Any, Dict
import mlflow
import mlflow.sklearn
import numpy as np

# Opt into MLflow file store backend for local tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class MLflowTracker:
    """Helper manager for initializing MLflow, logging parameters, metrics, artifacts, and models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MLflow tracking configuration.

        Args:
            config: MLflow configuration dictionary.
        """
        self.config = config
        self.tracking_uri = config.get("tracking_uri", "file:./mlruns")
        self.experiment_name = config.get("experiment_name", "fraud_detection_experiments")
        self.run_name = config.get("run_name", "baseline_smote_xgboost")

        # Set tracking URI and experiment
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str = None) -> mlflow.ActiveRun:
        """Start a new active MLflow run.

        Args:
            run_name: Optional custom run name override.

        Returns:
            Active MLflow run object.
        """
        active_name = run_name or self.run_name
        return mlflow.start_run(run_name=active_name)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters dictionary to active MLflow run.

        Args:
            params: Flat or nested dictionary of parameter keys and values.
        """
        flat_params = {}
        for key, val in params.items():
            if isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    flat_params[f"{key}_{sub_k}"] = str(sub_v)
            else:
                flat_params[key] = str(val)

        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log numerical metrics dictionary to active MLflow run.

        Args:
            metrics: Dictionary of metric names and numeric values.
        """
        numeric_metrics = {}

        def _extract(data: Dict[str, Any], prefix: str = ""):
            for k, v in data.items():
                name = f"{prefix}_{k}" if prefix else k
                if isinstance(v, (int, float, np.number)):
                    numeric_metrics[name] = float(v)
                elif isinstance(v, dict):
                    _extract(v, prefix=name)

        _extract(metrics)

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

    def log_artifacts(self, reports_dir: str):
        """Log files from reports directory as artifacts.

        Args:
            reports_dir: Path to directory containing plots and JSON reports.
        """
        if os.path.exists(reports_dir):
            mlflow.log_artifacts(reports_dir, artifact_path="evaluation_reports")

    def log_model(self, model: Any, artifact_path: str = "model"):
        """Log Scikit-learn / Imblearn pipeline model to MLflow.

        Args:
            model: Trained pipeline or model object.
            artifact_path: MLflow artifact subpath.
        """
        # Trust imblearn, xgboost, and sklearn internal types for skops serialization
        trusted_types = [
            "imblearn.over_sampling._smote.base.SMOTE",
            "imblearn.pipeline.Pipeline",
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
            "sklearn.neighbors._kd_tree.KDTree",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ]
        mlflow.sklearn.log_model(
            model, artifact_path=artifact_path, skops_trusted_types=trusted_types
        )

    def end_run(self):
        """Cleanly end active MLflow run."""
        mlflow.end_run()
