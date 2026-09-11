"""Controlled retraining workflow for candidate creation and promotion decisions."""

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator
from src.models.promotion import evaluate_predictions, evaluate_promotion, write_promotion_reports
from src.models.train import ModelTrainer


def run_retraining(
    new_data_path: str | Path,
    params_path: str | Path = "params.yaml",
    tracking_uri: str | None = None,
    promote: bool = False,
    report_dir: str | Path = "reports/model",
) -> dict[str, Any]:
    """Run candidate training/evaluation; promotion requires an explicit flag and passing gates."""
    params = yaml.safe_load(Path(params_path).read_text())
    raw = pd.read_csv(new_data_path)
    validation_config = params["data_validation"]
    validation = DataValidator(validation_config).validate(raw)
    if validation["overall_status"] != "PASSED":
        raise ValueError(f"New data failed validation: {validation}")
    if "isFraud" not in raw.columns:
        raise ValueError("Supervised retraining data must contain isFraud")

    prep_config = dict(params["preprocessing"])
    prep_config["raw_data_path"] = str(new_data_path)
    processed, _ = DataPreprocessor({**prep_config, "feature_engineering": params["feature_engineering"]}).process(raw)
    tracker_config = params["mlflow"]
    uri = tracking_uri or tracker_config["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    experiment_name = tracker_config.get("experiment_name", "fraud_detection_experiments")
    mlflow.set_experiment(experiment_name)
    trainer_config = {
        "processed_data_path": str(new_data_path),
        "target_column": "isFraud",
        "data_split": params["data_split"],
        "smote": params["smote"],
        "model": params["model"],
    }
    trainer = ModelTrainer(trainer_config)
    candidate, split = trainer.train(df=processed, use_smote=False, use_scale_pos_weight=True)
    threshold = float(params["model"]["threshold"])
    candidate_metrics = evaluate_predictions(candidate, split["X_test"], split["y_test"], threshold)

    with mlflow.start_run(run_name="retraining_candidate") as run:
        run_id = run.info.run_id
        mlflow.log_params({"training_dataset": str(new_data_path), "strategy": "scale_weight_only", "use_smote": False, "use_scale_pos_weight": True, "threshold": threshold})
        mlflow.log_metrics({key: value for key, value in candidate_metrics.items() if isinstance(value, (int, float))})
        mlflow.sklearn.log_model(candidate, name="model", serialization_format="cloudpickle")
    candidate_version = mlflow.register_model(f"runs:/{run_id}/model", tracker_config["registered_model"])
    loaded_candidate = mlflow.sklearn.load_model(
        f"models:/{tracker_config['registered_model']}/{candidate_version.version}"
    )
    candidate_metrics = evaluate_predictions(loaded_candidate, split["X_test"], split["y_test"], threshold)

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    champion_version = client.get_model_version_by_alias(tracker_config["registered_model"], tracker_config["champion_alias"])
    champion = mlflow.sklearn.load_model(f"models:/{tracker_config['registered_model']}@{tracker_config['champion_alias']}")
    champion_metrics = evaluate_predictions(champion, split["X_test"], split["y_test"], threshold)
    criteria = params.get("promotion", {})
    comparison = evaluate_promotion(criteria, candidate_metrics, champion_metrics)
    report = {
        "decision": comparison["decision"],
        "recommended_decision": comparison["decision"],
        "approval_state": "PENDING_APPROVAL" if comparison["decision"] == "PROMOTE" else "REJECTED",
        "candidate": {"run_id": run_id, "model_version": str(candidate_version.version), "metrics": candidate_metrics},
        "champion": {"run_id": champion_version.run_id, "model_version": str(champion_version.version), "metrics": champion_metrics},
        "comparison": comparison,
        "criteria": criteria,
        "failed_criteria": comparison["failed_criteria"],
        "reason": comparison["reason"],
    }
    if promote:
        raise ValueError("Automatic promotion is disabled; use scripts/approve_model.py with explicit APPROVE")
    write_promotion_reports(report, Path(report_dir) / "promotion_decision.json", Path(report_dir) / "promotion_decision.md")
    return report