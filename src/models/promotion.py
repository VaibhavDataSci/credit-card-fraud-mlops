"""Safety-gated candidate evaluation and MLflow model promotion."""

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(model: Any, X: Any, y: Any, threshold: float = 0.90) -> dict[str, Any]:
    """Evaluate a model at the deployment threshold without fitting it."""
    probabilities = np.asarray(model.predict_proba(X))
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == 0:
        classes = np.asarray(getattr(model.named_steps["clf"], "classes_", []))
    fraud_indices = np.flatnonzero(classes == 1)
    if fraud_indices.size != 1:
        raise ValueError("Model must expose exactly one class-1 fraud probability")
    fraud_probability = probabilities[:, fraud_indices[0]]
    if not np.isfinite(fraud_probability).all() or ((fraud_probability < 0) | (fraud_probability > 1)).any():
        raise ValueError("Model produced invalid probabilities")
    predictions = (fraud_probability >= threshold).astype(int)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(y, predictions, labels=[0, 1]).ravel()]
    return {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, fraud_probability)),
        "pr_auc": float(average_precision_score(y, fraud_probability)),
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        "test_samples": int(len(y)),
    }


def compare_metrics(candidate: dict[str, Any], champion: dict[str, Any]) -> dict[str, float]:
    metric_names = ("precision", "recall", "f1_score", "roc_auc", "pr_auc")
    return {f"{name}_delta": float(candidate[name] - champion[name]) for name in metric_names}


def evaluate_promotion(criteria: dict[str, float], candidate: dict[str, Any], champion: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative fraud-safety gates; no registry mutation occurs here."""
    deltas = compare_metrics(candidate, champion)
    checks = {
        "recall": deltas["recall_delta"] >= criteria.get("min_recall_delta", 0.0),
        "pr_auc": deltas["pr_auc_delta"] >= criteria.get("min_pr_auc_delta", 0.0),
        "f1_score": deltas["f1_score_delta"] >= criteria.get("min_f1_delta", 0.0),
        "false_negatives": candidate["confusion_matrix"]["false_negative"] <= champion["confusion_matrix"]["false_negative"],
        "candidate_metrics_valid": all(np.isfinite(candidate[name]) for name in ("precision", "recall", "f1_score", "roc_auc", "pr_auc")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"decision": "PROMOTE" if not failed else "REJECT", "checks": checks, "deltas": deltas, "failed_criteria": failed, "reason": "All promotion gates passed." if not failed else "Promotion rejected: " + ", ".join(failed)}


def write_promotion_reports(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    candidate, champion, comparison = report["candidate"], report["champion"], report["comparison"]
    lines = ["# Model Promotion Decision", "", f"Recommended Decision: **{report.get('recommended_decision', report['decision'])}**", f"Approval State: **{report.get('approval_state', 'PENDING_APPROVAL')}**", "", f"Reason: {report['reason']}", "", "## Candidate", "", f"Run ID: `{candidate.get('run_id')}`", f"Version: `{candidate.get('model_version')}`", "", "## Champion", "", f"Version: `{champion.get('model_version')}`", f"Run ID: `{champion.get('run_id')}`", "", "## Metrics", "", "| Metric | Candidate | Champion | Delta |", "|---|---:|---:|---:|"]
    for metric in ("precision", "recall", "f1_score", "roc_auc", "pr_auc"):
        lines.append(f"| {metric} | {candidate['metrics'][metric]:.6f} | {champion['metrics'][metric]:.6f} | {comparison['deltas'][metric + '_delta']:.6f} |")
    lines.extend(["", "## Criteria", ""])
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in comparison["checks"].items())
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote_candidate(tracking_uri: str, model_name: str, candidate_version: str, report: dict[str, Any]) -> None:
    """Assign champion only after a caller has recorded a PROMOTE decision."""
    if report.get("decision") != "PROMOTE" or report.get("approval_state") != "APPROVED":
        raise ValueError("Only a PROMOTE decision can change the Champion alias")
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(model_name, "champion", str(candidate_version))