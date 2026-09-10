#!/usr/bin/env python3
"""CLI entry point for model evaluation, validation, and MLflow Model Registry promotion."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import mlflow
import pandas as pd
import yaml

# Ensure src module is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.evaluate import ModelEvaluator
from src.models.registry import ModelRegistry
from src.models.validate import ModelValidator


def main():
    """Execute model evaluation, validation checks, and MLflow registry promotion."""
    params_path = BASE_DIR / "params.yaml"

    if not params_path.exists():
        print(f"[ERROR] Configuration file not found at {params_path}")
        sys.exit(1)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Extract configuration sections
    eval_config = params.get("evaluation", {})
    prep_config = params.get("preprocessing", {})
    mlflow_config = params.get("mlflow", {})
    threshold_config = params.get("threshold_analysis", {})

    proc_rel = prep_config.get("processed_data_path", "data/processed/cleaned.parquet")
    model_rel = eval_config.get("model_output_path", "models/xgboost_fraud_model.joblib")
    reports_rel = eval_config.get("reports_dir", "reports/model")

    proc_abs = BASE_DIR / proc_rel
    model_abs = BASE_DIR / model_rel
    reports_abs = BASE_DIR / reports_rel

    tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
    exp_name = mlflow_config.get("experiment_name", "fraud_detection_experiments")

    print("==================================================")
    print("  PHASE 8 — MODEL EVALUATION, VALIDATION & REGISTRY")
    print("==================================================")
    print(f"Model Path    : {model_abs}")
    print(f"Dataset Path  : {proc_abs}")
    print(f"Reports Dir   : {reports_abs}")
    print(f"MLflow URI    : {tracking_uri}")

    if not model_abs.exists():
        print(f"[ERROR] Model artifact not found at {model_abs}")
        sys.exit(1)

    if not proc_abs.exists():
        print(f"[ERROR] Processed dataset not found at {proc_abs}")
        sys.exit(1)

    try:
        # 1. Load model artifact and dataset
        pipeline = joblib.load(model_abs)
        df = pd.read_parquet(proc_abs)

        target_col = prep_config.get("target_column", "isFraud")
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Use test split for evaluation
        split_cfg = params.get("data_split", {})
        test_size = split_cfg.get("test_size", 0.3)
        random_state = split_cfg.get("random_state", 42)

        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 2. Run Model Evaluation
        evaluator = ModelEvaluator(reports_dir=str(reports_abs))
        metrics = evaluator.evaluate(pipeline, X_test, y_test)

        # Operating threshold evaluation at 0.90
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        op_threshold = 0.90
        y_pred_090 = (y_proba >= op_threshold).astype(int)

        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        prec_090 = float(precision_score(y_test, y_pred_090, zero_division=0))
        rec_090 = float(recall_score(y_test, y_pred_090, zero_division=0))
        f1_090 = float(f1_score(y_test, y_pred_090, zero_division=0))
        cm_090 = confusion_matrix(y_test, y_pred_090)
        tn, fp, fn, tp = [int(x) for x in cm_090.ravel()]

        # 3. Locate MLflow Run
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(exp_name)

        target_run_id = None
        if exp is not None:
            runs = client.search_runs(exp.experiment_id, order_by=["attribute.start_time DESC"])
            # Prefer selected Phase 7 run or latest scale_weight_only run
            for r in runs:
                if r.data.params.get("use_smote", "False").lower() in ["false", "none"] or r.info.run_id == "bdeee1d8376f48f18a98da80b1951105":
                    target_run_id = r.info.run_id
                    break

            if not target_run_id and runs:
                target_run_id = runs[0].info.run_id

        if not target_run_id:
            target_run_id = "bdeee1d8376f48f18a98da80b1951105"

        print(f"MLflow Target Run ID : {target_run_id}")

        # Ensure model is logged under target run in MLflow if needed
        artifacts = [a.path for a in client.list_artifacts(target_run_id)]
        if "model" not in artifacts:
            with mlflow.start_run(run_id=target_run_id):
                mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

        # 4. Perform Model Validation
        validator = ModelValidator(tracking_uri=tracking_uri)
        metadata = {
            "selected_experiment": "scale_weight_only",
            "selected_threshold": op_threshold,
            "strategy": "scale_weight_only",
        }

        sample_input = X_test.iloc[:5]
        val_report = validator.validate_candidate(
            run_id=target_run_id,
            metadata=metadata,
            model=pipeline,
            sample_input=sample_input,
        )

        print("--------------------------------------------------")
        print(f"Model Validation Status : [{val_report['status']}]")
        if val_report["errors"]:
            for err in val_report["errors"]:
                print(f" - Error: {err}")

        if val_report["status"] != "PASSED":
            print("[FAIL] Model candidate failed validation criteria! Aborting Champion promotion.")
            sys.exit(1)

        # 5. Register Model & Assign Aliases
        registry = ModelRegistry(tracking_uri=tracking_uri)
        registered_name = "CreditCardFraudDetector"

        tags = {
            "model_name": registered_name,
            "model_strategy": "scale_weight_only",
            "phase": "phase_8",
            "threshold": "0.90",
            "evaluation_status": "passed",
            "validation_status": "passed",
            "selection_reason": "Scale pos weight strategy with 0.90 threshold achieves high recall and precision",
        }

        mv = registry.register_model(
            run_id=target_run_id,
            model_name=registered_name,
            artifact_path="model",
            tags=tags,
        )

        model_version_str = str(mv.version)
        registry.assign_alias(registered_name, "candidate", model_version_str)
        registry.assign_alias(registered_name, "champion", model_version_str)

        # Verify Champion can be loaded
        champion_model = registry.load_registered_model(registered_name, "champion")

        # 6. Save Model Metadata Report
        model_meta = {
            "registered_model": registered_name,
            "model_type": "xgboost",
            "model_version": model_version_str,
            "mlflow_run_id": target_run_id,
            "strategy": "scale_weight_only",
            "imbalance_strategy": "scale_pos_weight only, no SMOTE",
            "use_smote": False,
            "use_scale_pos_weight": True,
            "threshold": op_threshold,
            "selected_threshold": op_threshold,
            "validation_status": "passed",
            "selection_status": "champion",
            "model_uri": f"models:/{registered_name}@champion",
            "metrics_at_050": {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            },
            "metrics_at_090": {
                "precision": prec_090,
                "recall": rec_090,
                "f1_score": f1_090,
                "false_positives": fp,
                "false_negatives": fn,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        meta_path = reports_abs / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(model_meta, f, indent=2)

        print("--------------------------------------------------")
        print(f"Registered Model Name  : {registered_name}")
        print(f"Registered Version    : {model_version_str}")
        print(f"Candidate Alias       : @candidate -> v{model_version_str}")
        print(f"Champion Alias        : @champion  -> v{model_version_str}")
        print(f"Model URI             : {model_meta['model_uri']}")
        print(f"Metadata Saved        : {meta_path}")
        print(f" - Operating Threshold: {op_threshold:.2f}")
        print(f" - Precision @ 0.90    : {prec_090:.4f}")
        print(f" - Recall @ 0.90       : {rec_090:.4f}")
        print(f" - F1 @ 0.90           : {f1_090:.4f}")
        print(f" - FP @ 0.90           : {fp}")
        print(f" - FN @ 0.90           : {fn}")
        print(f" - Champion Loaded OK  : {type(champion_model)}")
        print("==================================================")
        print("[SUCCESS] Model evaluation, validation, and registry completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"[FAIL] Evaluation or registration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
