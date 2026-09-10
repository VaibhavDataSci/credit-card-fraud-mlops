"""Model training module for Credit Card Fraud Detection pipeline."""

import os
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class ModelTrainer:
    """Trainer class for building, resampling, and fitting XGBoost fraud detection model."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.

        Args:
            config: Model training and pipeline configuration dictionary.
        """
        self.config = config
        self.processed_data_path = config.get("processed_data_path", "data/processed/cleaned.parquet")
        self.target_column = config.get("target_column", "isFraud")

        # Split settings
        split_cfg = config.get("data_split", {})
        self.test_size = split_cfg.get("test_size", 0.3)
        self.random_state = split_cfg.get("random_state", 42)
        self.stratify = split_cfg.get("stratify", True)

        # SMOTE settings
        smote_cfg = config.get("smote", {})
        self.smote_random_state = smote_cfg.get("random_state", 42)
        self.smote_k_neighbors = smote_cfg.get("k_neighbors", 5)

        # Model settings
        self.model_cfg = config.get("model", {})
        self.model_output_path = config.get("model_output_path", "models/xgboost_fraud_model.joblib")

    def load_data(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Load processed dataset and separate features and target.

        Args:
            df: Optional DataFrame to use directly.

        Returns:
            Tuple of (Features X, Target y).
        """
        if df is None:
            if not os.path.exists(self.processed_data_path):
                raise FileNotFoundError(f"Processed dataset not found at {self.processed_data_path}")
            df = pd.read_parquet(self.processed_data_path)

        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' missing from dataset")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform stratified train/test split.

        Args:
            X: Features DataFrame.
            y: Target Series.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        stratify_target = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_target,
        )

        return X_train, X_test, y_train, y_test

    def build_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_smote: bool = True,
        use_scale_pos_weight: bool = True,
    ) -> ImbPipeline:
        """Construct ImbPipeline wrapping ColumnTransformer, optional SMOTE, and XGBClassifier.

        Args:
            X_train: Training features DataFrame.
            y_train: Training target Series.
            use_smote: Whether to include SMOTE resampling step. Defaults to True.
            use_scale_pos_weight: Whether to compute and apply scale_pos_weight. Defaults to True.

        Returns:
            Configured ImbPipeline object.
        """
        # Separate column types
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()

        # Define column preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ]
        )

        # Calculate scale_pos_weight only when requested
        if use_scale_pos_weight:
            neg_count, pos_count = np.bincount(y_train)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        else:
            scale_pos_weight = 1.0

        # Define XGBoost Classifier
        xgb_clf = XGBClassifier(
            objective=self.model_cfg.get("objective", "binary:logistic"),
            eval_metric=self.model_cfg.get("eval_metric", "logloss"),
            n_estimators=self.model_cfg.get("n_estimators", 200),
            max_depth=self.model_cfg.get("max_depth", 6),
            learning_rate=self.model_cfg.get("learning_rate", 0.05),
            subsample=self.model_cfg.get("subsample", 0.8),
            colsample_bytree=self.model_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=self.model_cfg.get("random_state", 42),
            n_jobs=self.model_cfg.get("n_jobs", -1),
        )

        # Build pipeline steps conditionally
        steps = [("prep", preprocessor)]

        if use_smote:
            steps.append(
                (
                    "smote",
                    SMOTE(
                        random_state=self.smote_random_state,
                        k_neighbors=self.smote_k_neighbors,
                    ),
                )
            )

        steps.append(("clf", xgb_clf))

        pipeline = ImbPipeline(steps=steps)

        return pipeline

    def train(
        self,
        df: pd.DataFrame = None,
        use_smote: bool = None,
        use_scale_pos_weight: bool = None,
    ) -> Tuple[ImbPipeline, Dict[str, Any]]:
        """Execute full training pipeline.

        Args:
            df: Optional input DataFrame.
            use_smote: Whether to use SMOTE resampling. Defaults to config setting (default False).
            use_scale_pos_weight: Whether to compute scale_pos_weight. Defaults to config setting (default True).

        Returns:
            Tuple of (Trained ImbPipeline, Training metadata dictionary).
        """
        X, y = self.load_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        if use_smote is None:
            use_smote = self.model_cfg.get("use_smote", False)
        if use_scale_pos_weight is None:
            use_scale_pos_weight = self.model_cfg.get("use_scale_pos_weight", True)

        pipeline = self.build_pipeline(
            X_train, y_train, use_smote=use_smote, use_scale_pos_weight=use_scale_pos_weight
        )
        pipeline.fit(X_train, y_train)

        metadata = {
            "train_shape": {"rows": int(X_train.shape[0]), "columns": int(X_train.shape[1])},
            "test_shape": {"rows": int(X_test.shape[0]), "columns": int(X_test.shape[1])},
            "train_class_dist": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
            "test_class_dist": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
            "X_test": X_test,
            "y_test": y_test,
        }

        return pipeline, metadata

    def save_model(self, pipeline: ImbPipeline, output_path: str = None) -> str:
        """Serialize trained pipeline model artifact.

        Args:
            pipeline: Trained ImbPipeline object.
            output_path: Path to save serialized joblib file.

        Returns:
            Saved file path string.
        """
        save_path = output_path or self.model_output_path
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        joblib.dump(pipeline, save_path)
        return save_path
