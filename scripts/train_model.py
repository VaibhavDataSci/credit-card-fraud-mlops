#!/usr/bin/env python3
"""CLI entry point for model training and evaluation."""

import os
import sys
from pathlib import Path
import yaml

# Ensure src module is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer


def main():
    """Execute model training pipeline, save model artifact, and perform evaluation."""
    params_path = BASE_DIR / "params.yaml"

    if not params_path.exists():
        print(f"[ERROR] Configuration file not found at {params_path}")
        sys.exit(1)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Extract configuration sections
    model_config = params.get("model", {})
    split_config = params.get("data_split", {})
    smote_config = params.get("smote", {})
    eval_config = params.get("evaluation", {})
    prep_config = params.get("preprocessing", {})

    proc_rel = prep_config.get("processed_data_path", "data/processed/cleaned.parquet")
    model_rel = eval_config.get("model_output_path", "models/xgboost_fraud_model.joblib")
    reports_rel = eval_config.get("reports_dir", "reports/model")

    proc_abs = BASE_DIR / proc_rel
    model_abs = BASE_DIR / model_rel
    reports_abs = BASE_DIR / reports_rel

    train_config = {
        "processed_data_path": str(proc_abs),
        "target_column": prep_config.get("target_column", "isFraud"),
        "data_split": split_config,
        "smote": smote_config,
        "model": model_config,
        "model_output_path": str(model_abs),
    }

    print("==================================================")
    print("      RUNNING MODEL TRAINING & EVALUATION PIPELINE")
    print("==================================================")
    print(f"Processed Data Path : {proc_abs}")
    print(f"Model Output Path   : {model_abs}")
    print(f"Reports Output Path : {reports_abs}")

    try:
        # 1. Train Model
        trainer = ModelTrainer(train_config)
        pipeline, metadata = trainer.train()
        saved_model_path = trainer.save_model(pipeline)
        model_size_mb = round(os.path.getsize(saved_model_path) / (1024 * 1024), 2)

        print("--------------------------------------------------")
        print(f"[SUCCESS] Model Training Completed!")
        print(f" - Model Artifact Saved: {saved_model_path} ({model_size_mb} MB)")
        print(f" - Train Shape         : {metadata['train_shape']['rows']} rows, {metadata['train_shape']['columns']} features")
        print(f" - Test Shape          : {metadata['test_shape']['rows']} rows, {metadata['test_shape']['columns']} features")
        print(f" - Train Class Balance : {metadata['train_class_dist']}")
        print(f" - Test Class Balance  : {metadata['test_class_dist']}")

        # 2. Evaluate Model on Untouched Test Set
        evaluator = ModelEvaluator(reports_dir=str(reports_abs))
        metrics = evaluator.evaluate(pipeline, metadata["X_test"], metadata["y_test"])

        print("--------------------------------------------------")
        print("          TEST SET EVALUATION RESULTS             ")
        print("--------------------------------------------------")
        print(f" - Test Samples        : {metrics['test_samples']:,}")
        print(f" - Precision           : {metrics['precision']:.4f}")
        print(f" - Recall              : {metrics['recall']:.4f}")
        print(f" - F1 Score            : {metrics['f1_score']:.4f}")
        print(f" - ROC-AUC             : {metrics['roc_auc']:.4f}")
        print(f" - PR-AUC (Average P)  : {metrics['pr_auc']:.4f}")
        print(f" - Confusion Matrix    : TN={metrics['confusion_matrix']['true_negative']:,}, FP={metrics['confusion_matrix']['false_positive']:,}, FN={metrics['confusion_matrix']['false_negative']:,}, TP={metrics['confusion_matrix']['true_positive']:,}")
        print("==================================================")
        print("[SUCCESS] Model training and evaluation pipeline completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"[FAIL] Model training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
