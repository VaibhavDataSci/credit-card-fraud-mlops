"""Model validation module for verifying deployment candidate models prior to Champion promotion."""

import os
from typing import Any, Dict, List, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class ModelValidator:
    """Validator for enforcing performance, artifact integrity, and schema expectations on candidate models."""

    REQUIRED_METRICS = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    REQUIRED_STRATEGY = "scale_weight_only"
    REQUIRED_THRESHOLD = 0.90

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """Initialize ModelValidator with tracking URI.

        Args:
            tracking_uri: MLflow tracking URI.
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def validate_run(self, run_id: str) -> Tuple[bool, List[str]]:
        """Validate MLflow run existence, parameters, metrics, and artifact presence.

        Args:
            run_id: MLflow run ID string.

        Returns:
            Tuple of (is_valid: bool, error_messages: List[str]).
        """
        errors = []
        try:
            run = self.client.get_run(run_id)
        except Exception as e:
            return False, [f"MLflow run '{run_id}' does not exist: {str(e)}"]

        params = run.data.params
        metrics = run.data.metrics

        # Check required metrics
        for metric_name in self.REQUIRED_METRICS:
            if metric_name not in metrics:
                errors.append(f"Required metric '{metric_name}' missing from MLflow run.")

        # Check strategy
        exp_name = params.get("experiment_name") or params.get("imbalance_strategy") or ""
        use_smote = params.get("use_smote", "True")
        use_scale = params.get("use_scale_pos_weight", "True")

        if use_smote.lower() == "true" or use_scale.lower() != "true":
            errors.append(
                f"Model strategy violates expected '{self.REQUIRED_STRATEGY}'. "
                f"Found use_smote={use_smote}, use_scale_pos_weight={use_scale}."
            )

        # Check model artifact presence
        has_model = False
        try:
            mlflow.models.get_model_info(f"runs:/{run_id}/model")
            has_model = True
        except Exception:
            artifacts = [a.path for a in self.client.list_artifacts(run_id)]
            has_model = "model" in artifacts or "evaluation_reports" in artifacts or len(artifacts) > 0

        if not has_model:
            errors.append(f"No valid model artifact path found in MLflow run '{run_id}'.")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_model_artifact(self, model: Any, sample_input: pd.DataFrame = None) -> Tuple[bool, List[str]]:
        """Validate that the model artifact is loadable and can generate valid predictions.

        Args:
            model: Loaded model or pipeline object.
            sample_input: Representative DataFrame containing required features.

        Returns:
            Tuple of (is_valid: bool, error_messages: List[str]).
        """
        errors = []
        if model is None:
            return False, ["Model object is None."]

        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            return False, ["Loaded model object missing predict() or predict_proba() interface."]

        if sample_input is not None:
            try:
                probas = model.predict_proba(sample_input)
                preds = model.predict(sample_input)

                if probas.shape[0] != len(sample_input):
                    errors.append("Prediction probability output shape mismatch.")
                if preds.shape[0] != len(sample_input):
                    errors.append("Prediction output shape mismatch.")

                # Ensure non-NaN output
                if np.isnan(probas).any() or np.isnan(preds).any():
                    errors.append("Model output contains NaN values.")

            except Exception as e:
                errors.append(f"Failed to generate prediction on representative transaction: {str(e)}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_candidate(
        self,
        run_id: str,
        metadata: Dict[str, Any],
        model: Any = None,
        sample_input: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive validation suite on candidate model prior to Champion promotion.

        Args:
            run_id: MLflow run ID string.
            metadata: Model metadata dictionary.
            model: Loaded model instance.
            sample_input: Representative input transaction.

        Returns:
            Validation report dictionary.
        """
        all_errors = []

        # 1. MLflow Run Validation
        run_valid, run_errors = self.validate_run(run_id)
        all_errors.extend(run_errors)

        # 2. Threshold & Metadata Validation
        threshold = metadata.get("selected_threshold") or metadata.get("threshold", 0.0)
        if abs(threshold - self.REQUIRED_THRESHOLD) > 1e-4:
            all_errors.append(f"Selected threshold ({threshold}) does not match expected {self.REQUIRED_THRESHOLD}.")

        strategy = metadata.get("selected_experiment") or metadata.get("strategy", "")
        if strategy != self.REQUIRED_STRATEGY:
            all_errors.append(f"Selected strategy '{strategy}' does not match expected '{self.REQUIRED_STRATEGY}'.")

        # 3. Model Artifact & Inference Validation
        if model is not None:
            art_valid, art_errors = self.validate_model_artifact(model, sample_input)
            all_errors.extend(art_errors)

        status = "PASSED" if len(all_errors) == 0 else "FAILED"
        return {
            "status": status,
            "run_id": run_id,
            "strategy": strategy,
            "threshold": threshold,
            "errors": all_errors,
        }
