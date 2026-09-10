"""Load and serve the validated Champion model from MLflow."""

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ChampionModelLoader:
    """Singleton-style loader for the MLflow Champion model."""

    def __init__(self, params_path: str | Path | None = None):
        root = Path(__file__).resolve().parent.parent
        config_path = Path(params_path) if params_path else root / "params.yaml"
        with config_path.open() as params_file:
            params = yaml.safe_load(params_file) or {}

        mlflow_config = params.get("mlflow", {})
        model_config = params.get("model", {})
        self.model_name = mlflow_config.get("registered_model", "CreditCardFraudDetector")
        self.alias = mlflow_config.get("champion_alias", "champion")
        self.model_uri = f"models:/{self.model_name}@{self.alias}"
        self.tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
        self.strategy = model_config.get("strategy")
        self.threshold = float(model_config.get("threshold", 0.90))
        self.model: Any = None
        self.version: str | None = None
        self.run_id: str | None = None
        self.loaded = False

    def load(self) -> None:
        """Load the Champion once and resolve its registry metadata."""
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient(tracking_uri=self.tracking_uri)
            model_version = client.get_model_version_by_alias(self.model_name, self.alias)
            self.model = mlflow.sklearn.load_model(self.model_uri)
            self.version = str(model_version.version)
            self.run_id = model_version.run_id
            self.strategy = model_version.tags.get("model_strategy", self.strategy)
            self.threshold = float(model_version.tags.get("threshold", self.threshold))
            self.loaded = True
            logger.info("Loaded MLflow model %s version %s", self.model_uri, self.version)
        except Exception:
            self.model = None
            self.loaded = False
            logger.exception("Unable to load MLflow Champion model %s", self.model_uri)
            raise

    def predict(self, transaction: dict[str, Any]) -> float:
        """Engineer one transaction and return the probability of class 1."""
        if not self.loaded or self.model is None:
            raise RuntimeError("Champion model is unavailable")

        features = FeatureEngineer({"enable_balance_diffs": True, "enable_amount_ratio": True}).transform(
            pd.DataFrame([transaction])
        )
        probabilities = self.model.predict_proba(features)
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            classes = getattr(self.model.named_steps["clf"], "classes_", None)
        fraud_indices = [index for index, value in enumerate(classes) if int(value) == 1]
        if not fraud_indices:
            raise RuntimeError("Loaded model does not expose a fraud class")
        return float(probabilities[0, fraud_indices[0]])

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "alias": self.alias,
            "model_uri": self.model_uri,
            "version": self.version,
            "run_id": self.run_id,
            "strategy": self.strategy,
            "threshold": self.threshold,
        }
