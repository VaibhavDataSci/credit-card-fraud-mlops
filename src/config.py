"""Centralized configuration loading module for environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


class Config:
    """Application configuration settings."""

    # Application Settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "credit-card-fraud-detection"
    )
    MLFLOW_REGISTERED_MODEL_NAME: str = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME", "CreditCardFraudDetector"
    )

    # Data Settings
    DATA_RAW_PATH: Path = BASE_DIR / os.getenv("DATA_RAW_PATH", "data/raw/creditcard.csv")
    DATA_PROCESSED_PATH: Path = BASE_DIR / os.getenv(
        "DATA_PROCESSED_PATH", "data/processed/cleaned.csv"
    )

    # Monitoring Settings
    DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.05"))


config = Config()
