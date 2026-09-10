"""MLflow Model Registry manager module for model lifecycle management."""

import os
from typing import Any, Dict, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Opt into MLflow file store backend for local tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class ModelRegistry:
    """Helper manager for registering models, setting tags, managing aliases, and loading models."""

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """Initialize registry manager with tracking URI.

        Args:
            tracking_uri: MLflow tracking server or file store URI.
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model(
        self,
        run_id: str,
        model_name: str = "CreditCardFraudDetector",
        artifact_path: str = "model",
        tags: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Register a model artifact from an MLflow run into the Model Registry.

        Args:
            run_id: MLflow run ID containing the trained model artifact.
            model_name: Name of the registered model.
            artifact_path: Subpath of the model artifact within the run.
            tags: Optional dictionary of tags to record on the registered version.

        Returns:
            Registered ModelVersion object.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_version = mlflow.register_model(model_uri, model_name)

        if tags:
            for key, val in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=str(val),
                )

        return model_version

    def assign_alias(self, model_name: str, alias: str, version: str) -> None:
        """Assign an alias (e.g. 'candidate', 'champion') to a specific registered model version.

        Args:
            model_name: Registered model name.
            alias: Alias string to assign.
            version: Model version number or string.
        """
        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version),
        )

    def get_model_uri(self, model_name: str = "CreditCardFraudDetector", alias_or_version: str = "champion") -> str:
        """Construct MLflow model URI for a given alias or version.

        Args:
            model_name: Registered model name.
            alias_or_version: Alias (e.g., 'champion') or explicit version number.

        Returns:
            MLflow model URI string.
        """
        if alias_or_version.isdigit():
            return f"models:/{model_name}/{alias_or_version}"
        else:
            return f"models:/{model_name}@{alias_or_version}"

    def load_registered_model(
        self, model_name: str = "CreditCardFraudDetector", alias_or_version: str = "champion"
    ) -> Any:
        """Load registered model from MLflow registry using alias or version.

        Args:
            model_name: Registered model name.
            alias_or_version: Alias name (e.g., 'champion') or version number string.

        Returns:
            Loaded model object.
        """
        model_uri = self.get_model_uri(model_name, alias_or_version)
        return mlflow.sklearn.load_model(model_uri)
