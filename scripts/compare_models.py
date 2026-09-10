#!/usr/bin/env python3
"""CLI entry point for model comparison, threshold analysis, and deployment candidate selection."""

import os
import sys
from pathlib import Path

import yaml

# Ensure src module is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.compare import (
    ModelSelector,
    ThresholdAnalyzer,
    plot_experiment_comparison,
)
from src.models.evaluate import ModelEvaluator
from src.models.tracking import MLflowTracker
from src.models.train import ModelTrainer


def main():
    """Execute model comparison pipeline across imbalance strategies."""
    params_path = BASE_DIR / "params.yaml"

    if not params_path.exists():
        print(f"[ERROR] Configuration file not found at {params_path}")
        sys.exit(1)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Extract shared configuration sections
    model_config = params.get("model", {})
    split_config = params.get("data_split", {})
    smote_config = params.get("smote", {})
    eval_config = params.get("evaluation", {})
    prep_config = params.get("preprocessing", {})
    mlflow_config = params.get("mlflow", {})
    experiments_config = params.get("experiments", {})
    threshold_config = params.get("threshold_analysis", {})

    proc_rel = prep_config.get("processed_data_path", "data/processed/cleaned.parquet")
    reports_rel = eval_config.get("reports_dir", "reports/model")
    proc_abs = BASE_DIR / proc_rel
    reports_abs = str(BASE_DIR / reports_rel)

    if not proc_abs.exists():
        print(f"[ERROR] Processed dataset not found at {proc_abs}")
        print("Run 'python scripts/preprocess_data.py' first.")
        sys.exit(1)

    print("==================================================")
    print("  PHASE 7 — MODEL COMPARISON & SELECTION PIPELINE")
    print("==================================================")
    print(f"Processed Data : {proc_abs}")
    print(f"Reports Dir    : {reports_abs}")
    print(f"Experiments    : {list(experiments_config.keys())}")
    print()

    # ----------------------------------------------------------------
    # 1. Load data ONCE and split ONCE for consistency
    # ----------------------------------------------------------------
    base_train_config = {
        "processed_data_path": str(proc_abs),
        "target_column": prep_config.get("target_column", "isFraud"),
        "data_split": split_config,
        "smote": smote_config,
        "model": model_config,
        "model_output_path": str(BASE_DIR / "models" / "xgboost_fraud_model.joblib"),
    }

    trainer = ModelTrainer(base_train_config)
    X, y = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    print(f"Dataset Split  : Train={X_train.shape[0]:,} rows, Test={X_test.shape[0]:,} rows")
    print()

    # ----------------------------------------------------------------
    # 2. Run each experiment
    # ----------------------------------------------------------------
    experiment_results = {}
    experiment_pipelines = {}
    experiment_probas = {}
    experiment_run_ids = {}

    tracker = MLflowTracker(mlflow_config)

    for exp_name, exp_cfg in experiments_config.items():
        use_smote = exp_cfg.get("use_smote", True)
        use_scale = exp_cfg.get("use_scale_pos_weight", True)
        description = exp_cfg.get("description", exp_name)

        print("--------------------------------------------------")
        print(f"  EXPERIMENT: {exp_name}")
        print(f"  Strategy  : {description}")
        print(f"  SMOTE     : {use_smote}")
        print(f"  ScalePosW : {use_scale}")
        print("--------------------------------------------------")

        # Build and train pipeline
        pipeline = trainer.build_pipeline(
            X_train, y_train,
            use_smote=use_smote,
            use_scale_pos_weight=use_scale,
        )
        pipeline.fit(X_train, y_train)

        # Evaluate on untouched test set
        evaluator = ModelEvaluator(reports_dir=reports_abs)
        metrics = evaluator.evaluate(pipeline, X_test, y_test)
        metrics["strategy"] = description

        # Get predicted probabilities for threshold analysis later
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Store results
        experiment_results[exp_name] = metrics
        experiment_pipelines[exp_name] = pipeline
        experiment_probas[exp_name] = y_proba

        # Log to MLflow
        run = tracker.start_run(run_name=exp_name)
        run_id = run.run_id if hasattr(run, "run_id") else str(run)

        # Access run_id from the active run context
        import mlflow
        run_id = mlflow.active_run().info.run_id

        experiment_run_ids[exp_name] = run_id

        # Log parameters
        tracker.log_params({
            "experiment_name": exp_name,
            "imbalance_strategy": description,
            "use_smote": str(use_smote),
            "use_scale_pos_weight": str(use_scale),
            "model_type": "xgboost",
            "data_split": split_config,
            "smote": smote_config if use_smote else {"enabled": False},
            "model": model_config,
        })

        # Log metrics including confusion matrix values
        cm = metrics.get("confusion_matrix", {})
        tracker.log_metrics({
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "true_positives": cm.get("true_positive", 0),
            "true_negatives": cm.get("true_negative", 0),
            "false_positives": cm.get("false_positive", 0),
            "false_negatives": cm.get("false_negative", 0),
        })

        tracker.log_artifacts(reports_abs)
        tracker.log_model(pipeline, artifact_path="model")
        tracker.end_run()

        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
        print(f"  FP={cm.get('false_positive', 0):,}  FN={cm.get('false_negative', 0):,}")
        print(f"  MLflow Run: {run_id}")
        print()

    # ----------------------------------------------------------------
    # 3. Select best experiment
    # ----------------------------------------------------------------
    print("==================================================")
    print("  MODEL SELECTION")
    print("==================================================")

    selector = ModelSelector(min_recall=0.95)
    selected_name, selected_metrics = selector.select_best_experiment(experiment_results)

    print(f"  Selected Experiment: {selected_name}")
    print(f"  Strategy           : {selected_metrics.get('strategy', 'N/A')}")
    print(f"  MLflow Run ID      : {experiment_run_ids[selected_name]}")
    print()

    # ----------------------------------------------------------------
    # 4. Threshold analysis on selected model
    # ----------------------------------------------------------------
    print("==================================================")
    print("  THRESHOLD ANALYSIS")
    print("==================================================")

    thresholds = threshold_config.get("thresholds", [0.50, 0.60, 0.70, 0.80, 0.90])
    analyzer = ThresholdAnalyzer(thresholds=thresholds)
    threshold_results = analyzer.analyze(y_test.values, experiment_probas[selected_name])

    for t in threshold_results:
        print(
            f"  Threshold={t['threshold']:.2f}  "
            f"P={t['precision']:.4f}  R={t['recall']:.4f}  "
            f"F1={t['f1_score']:.4f}  "
            f"FP={t['false_positives']:,}  FN={t['false_negatives']:,}"
        )

    # Select best threshold
    best_threshold = selector.select_best_threshold(threshold_results)

    print()
    print(f"  Selected Threshold : {best_threshold['threshold']:.2f}")
    print(f"  Precision          : {best_threshold['precision']:.4f}")
    print(f"  Recall             : {best_threshold['recall']:.4f}")
    print(f"  F1 Score           : {best_threshold['f1_score']:.4f}")
    print()

    # ----------------------------------------------------------------
    # 5. Build selection reasoning
    # ----------------------------------------------------------------
    # Compare all experiments to build reasoning text
    all_names = list(experiment_results.keys())
    reasoning_parts = []

    reasoning_parts.append(
        f"Three imbalance strategies were compared: {', '.join(all_names)}."
    )

    for name, m in experiment_results.items():
        cm_data = m.get("confusion_matrix", {})
        reasoning_parts.append(
            f"- **{name}**: Recall={m['recall']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
            f"Precision={m['precision']:.4f}, F1={m['f1_score']:.4f}, "
            f"FP={cm_data.get('false_positive', 0):,}, FN={cm_data.get('false_negative', 0):,}"
        )

    sel_cm = selected_metrics.get("confusion_matrix", {})
    reasoning_parts.append("")
    reasoning_parts.append(
        f"**{selected_name}** was selected because it achieved the best combination of "
        f"Recall ({selected_metrics['recall']:.4f}) and PR-AUC ({selected_metrics['pr_auc']:.4f}) "
        f"while maintaining {sel_cm.get('false_negative', 0):,} false negatives. "
    )

    # Add threshold reasoning
    reasoning_parts.append("")
    reasoning_parts.append(
        f"Threshold {best_threshold['threshold']:.2f} was selected because it "
        f"achieves Recall={best_threshold['recall']:.4f} (≥ 0.95) "
        f"with F1={best_threshold['f1_score']:.4f}, balancing fraud detection "
        f"against {best_threshold['false_positives']:,} false positives."
    )

    selection_reasoning = "\n".join(reasoning_parts)

    # ----------------------------------------------------------------
    # 6. Generate reports and metadata
    # ----------------------------------------------------------------
    print("==================================================")
    print("  GENERATING REPORTS")
    print("==================================================")

    # Comparison report JSON
    comparison_report = selector.generate_comparison_report(
        experiment_results=experiment_results,
        selected_experiment=selected_name,
        threshold_results=threshold_results,
        selected_threshold=best_threshold,
        reports_dir=reports_abs,
    )
    print(f"  ✓ experiment_comparison.json")

    # Model metadata JSON
    metadata = selector.generate_metadata(
        selected_experiment=selected_name,
        experiment_config=experiments_config[selected_name],
        metrics=selected_metrics,
        selected_threshold=best_threshold,
        mlflow_run_id=experiment_run_ids[selected_name],
        mlflow_experiment_name=mlflow_config.get("experiment_name", "fraud_detection_experiments"),
        reports_dir=reports_abs,
    )
    print(f"  ✓ model_metadata.json")

    # Human-readable selection report
    selector.generate_selection_report(
        experiment_results=experiment_results,
        selected_experiment=selected_name,
        threshold_results=threshold_results,
        selected_threshold=best_threshold,
        selection_reasoning=selection_reasoning,
        reports_dir=reports_abs,
    )
    print(f"  ✓ model_selection.md")

    # Visualization
    plot_experiment_comparison(
        experiment_results=experiment_results,
        selected_experiment=selected_name,
        reports_dir=reports_abs,
    )
    print(f"  ✓ experiment_comparison.png")

    # ----------------------------------------------------------------
    # 7. Save the selected model artifact locally
    # ----------------------------------------------------------------
    model_output = str(BASE_DIR / "models" / "xgboost_fraud_model.joblib")
    saved_path = trainer.save_model(experiment_pipelines[selected_name], output_path=model_output)
    model_size_mb = round(os.path.getsize(saved_path) / (1024 * 1024), 2)
    print(f"  ✓ Model artifact saved: {saved_path} ({model_size_mb} MB)")

    print()
    print("==================================================")
    print("  PHASE 7 COMPLETE — DEPLOYMENT CANDIDATE SELECTED")
    print("==================================================")
    print(f"  Experiment : {selected_name}")
    print(f"  Threshold  : {best_threshold['threshold']:.2f}")
    print(f"  Run ID     : {experiment_run_ids[selected_name]}")
    print(f"  Precision  : {best_threshold['precision']:.4f}")
    print(f"  Recall     : {best_threshold['recall']:.4f}")
    print(f"  F1 Score   : {best_threshold['f1_score']:.4f}")
    print("==================================================")

    sys.exit(0)


if __name__ == "__main__":
    main()
