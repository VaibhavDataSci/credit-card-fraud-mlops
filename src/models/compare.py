"""Model comparison, threshold analysis, and selection module for fraud detection pipeline."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ThresholdAnalyzer:
    """Evaluate model predictions across multiple classification thresholds."""

    def __init__(self, thresholds: List[float] = None):
        """Initialize with threshold grid.

        Args:
            thresholds: List of probability thresholds to evaluate.
        """
        self.thresholds = thresholds or [0.50, 0.60, 0.70, 0.80, 0.90]

    def analyze(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Evaluate metrics at each threshold.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            List of dicts, one per threshold, containing threshold-specific metrics.
        """
        results = []
        for threshold in self.thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = [int(x) for x in cm.ravel()]

            results.append({
                "threshold": threshold,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
            })

        return results


class ModelSelector:
    """Compare experiment results and select the best deployment candidate."""

    def __init__(self, min_recall: float = 0.95):
        """Initialize selector with minimum recall constraint.

        Args:
            min_recall: Minimum acceptable recall for a deployment candidate.
        """
        self.min_recall = min_recall

    def select_best_experiment(
        self, experiment_results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Select the best experiment based on business-oriented criteria.

        Selection criteria (applied in order):
        1. Filter experiments with recall >= min_recall.
        2. Among qualifying experiments, pick highest PR-AUC.
        3. Tiebreaker: lowest false negatives, then highest F1.

        If no experiment meets min_recall, pick the one with highest recall.

        Args:
            experiment_results: Dict mapping experiment name to its metrics dict.

        Returns:
            Tuple of (selected experiment name, its metrics dict).
        """
        # Separate qualifying (recall >= threshold) from non-qualifying
        qualifying = {
            name: metrics
            for name, metrics in experiment_results.items()
            if metrics.get("recall", 0) >= self.min_recall
        }

        if qualifying:
            candidates = qualifying
        else:
            # Fallback: pick highest recall regardless
            candidates = experiment_results

        # Sort by: PR-AUC desc, false_negatives asc, F1 desc
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: (
                item[1].get("pr_auc", 0),
                -item[1].get("confusion_matrix", {}).get("false_negative", float("inf")),
                item[1].get("f1_score", 0),
            ),
            reverse=True,
        )

        best_name, best_metrics = sorted_candidates[0]
        return best_name, best_metrics

    def select_best_threshold(
        self, threshold_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select the best prediction threshold from analysis results.

        Criteria: Among thresholds with recall >= min_recall, pick highest F1.
        If none qualify, pick the threshold with highest recall.

        Args:
            threshold_results: List of threshold analysis result dicts.

        Returns:
            The selected threshold result dict.
        """
        qualifying = [
            t for t in threshold_results if t.get("recall", 0) >= self.min_recall
        ]

        if qualifying:
            # Pick highest F1 among qualifying thresholds
            return max(qualifying, key=lambda t: t.get("f1_score", 0))
        else:
            # Fallback: highest recall
            return max(threshold_results, key=lambda t: t.get("recall", 0))

    def generate_comparison_report(
        self,
        experiment_results: Dict[str, Dict[str, Any]],
        selected_experiment: str,
        threshold_results: List[Dict[str, Any]],
        selected_threshold: Dict[str, Any],
        reports_dir: str,
    ) -> Dict[str, Any]:
        """Generate structured comparison report as JSON.

        Args:
            experiment_results: All experiment metrics.
            selected_experiment: Name of the selected experiment.
            threshold_results: Threshold analysis results.
            selected_threshold: The selected threshold result.
            reports_dir: Directory to save reports.

        Returns:
            The full comparison report dictionary.
        """
        os.makedirs(reports_dir, exist_ok=True)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiments": {},
            "selected_experiment": selected_experiment,
            "threshold_analysis": threshold_results,
            "selected_threshold": selected_threshold["threshold"],
            "selection_criteria": {
                "primary": "Recall >= 0.95 AND highest PR-AUC",
                "secondary": "Lowest false negatives",
                "tiebreaker": "Highest F1 Score",
            },
        }

        for name, metrics in experiment_results.items():
            report["experiments"][name] = {
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "roc_auc": metrics.get("roc_auc", 0),
                "pr_auc": metrics.get("pr_auc", 0),
                "false_positives": metrics.get("confusion_matrix", {}).get("false_positive", 0),
                "false_negatives": metrics.get("confusion_matrix", {}).get("false_negative", 0),
                "true_positives": metrics.get("confusion_matrix", {}).get("true_positive", 0),
                "true_negatives": metrics.get("confusion_matrix", {}).get("true_negative", 0),
            }

        # Save JSON report
        report_path = os.path.join(reports_dir, "experiment_comparison.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def generate_metadata(
        self,
        selected_experiment: str,
        experiment_config: Dict[str, Any],
        metrics: Dict[str, Any],
        selected_threshold: Dict[str, Any],
        mlflow_run_id: str,
        mlflow_experiment_name: str,
        reports_dir: str,
    ) -> Dict[str, Any]:
        """Generate deployment candidate metadata JSON.

        Args:
            selected_experiment: Name of the selected experiment.
            experiment_config: Configuration of the selected experiment.
            metrics: Metrics of the selected experiment at default threshold.
            selected_threshold: Threshold analysis result for the chosen threshold.
            mlflow_run_id: MLflow run ID of the selected experiment.
            mlflow_experiment_name: MLflow experiment name.
            reports_dir: Directory to save metadata.

        Returns:
            The metadata dictionary.
        """
        os.makedirs(reports_dir, exist_ok=True)

        metadata = {
            "model_name": "CreditCardFraudDetector",
            "model_type": "xgboost",
            "experiment_name": mlflow_experiment_name,
            "mlflow_run_id": mlflow_run_id,
            "selected_experiment": selected_experiment,
            "imbalance_strategy": experiment_config.get("description", ""),
            "use_smote": experiment_config.get("use_smote", True),
            "use_scale_pos_weight": experiment_config.get("use_scale_pos_weight", True),
            "selected_threshold": selected_threshold["threshold"],
            "evaluation_metrics": {
                "at_default_threshold": {
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "roc_auc": metrics.get("roc_auc", 0),
                    "pr_auc": metrics.get("pr_auc", 0),
                },
                "at_selected_threshold": {
                    "precision": selected_threshold.get("precision", 0),
                    "recall": selected_threshold.get("recall", 0),
                    "f1_score": selected_threshold.get("f1_score", 0),
                    "false_positives": selected_threshold.get("false_positives", 0),
                    "false_negatives": selected_threshold.get("false_negatives", 0),
                },
            },
            "selection_date": datetime.now(timezone.utc).isoformat(),
        }

        metadata_path = os.path.join(reports_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def generate_selection_report(
        self,
        experiment_results: Dict[str, Dict[str, Any]],
        selected_experiment: str,
        threshold_results: List[Dict[str, Any]],
        selected_threshold: Dict[str, Any],
        selection_reasoning: str,
        reports_dir: str,
    ):
        """Generate human-readable model selection markdown report.

        Args:
            experiment_results: All experiment metrics.
            selected_experiment: Name of selected experiment.
            threshold_results: Threshold analysis results.
            selected_threshold: Selected threshold dict.
            selection_reasoning: Free-text reasoning string.
            reports_dir: Directory to save the report.
        """
        os.makedirs(reports_dir, exist_ok=True)

        lines = [
            "# Model Selection Report",
            "",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "---",
            "",
            "## Experiment Comparison",
            "",
            "| Experiment | Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |",
            "|------------|----------|-----------|--------|------|---------|--------|------|------|",
        ]

        for name, m in experiment_results.items():
            cm = m.get("confusion_matrix", {})
            lines.append(
                f"| {name} | {m.get('strategy', 'N/A')} | "
                f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | "
                f"{m.get('f1_score', 0):.4f} | {m.get('roc_auc', 0):.4f} | "
                f"{m.get('pr_auc', 0):.4f} | "
                f"{cm.get('false_positive', 0):,} | {cm.get('false_negative', 0):,} |"
            )

        lines += [
            "",
            "---",
            "",
            "## Threshold Analysis (Selected Experiment)",
            "",
            "| Threshold | Precision | Recall | F1 | FP | FN |",
            "|-----------|-----------|--------|------|------|------|",
        ]

        for t in threshold_results:
            lines.append(
                f"| {t['threshold']:.2f} | {t['precision']:.4f} | "
                f"{t['recall']:.4f} | {t['f1_score']:.4f} | "
                f"{t['false_positives']:,} | {t['false_negatives']:,} |"
            )

        lines += [
            "",
            "---",
            "",
            "## Selected Model",
            "",
            f"- **Experiment**: `{selected_experiment}`",
            f"- **Selected Threshold**: `{selected_threshold['threshold']:.2f}`",
            f"- **Precision at threshold**: `{selected_threshold.get('precision', 0):.4f}`",
            f"- **Recall at threshold**: `{selected_threshold.get('recall', 0):.4f}`",
            f"- **F1 at threshold**: `{selected_threshold.get('f1_score', 0):.4f}`",
            "",
            "---",
            "",
            "## Selection Reasoning",
            "",
            selection_reasoning,
            "",
        ]

        report_path = os.path.join(reports_dir, "model_selection.md")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))


def plot_experiment_comparison(
    experiment_results: Dict[str, Dict[str, Any]],
    selected_experiment: str,
    reports_dir: str,
):
    """Create a grouped bar chart comparing key metrics across experiments.

    Args:
        experiment_results: All experiment metrics.
        selected_experiment: Name of the selected experiment (highlighted).
        reports_dir: Directory to save the plot.
    """
    os.makedirs(reports_dir, exist_ok=True)

    names = list(experiment_results.keys())
    metrics_to_plot = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]

    x = np.arange(len(names))
    width = 0.15
    offsets = np.arange(len(metrics_to_plot)) - len(metrics_to_plot) / 2 + 0.5

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        values = [experiment_results[n].get(metric, 0) for n in names]
        ax.bar(x + offsets[i] * width, values, width, label=label, color=color, alpha=0.85)

    # Highlight selected experiment
    selected_idx = names.index(selected_experiment) if selected_experiment in names else -1
    if selected_idx >= 0:
        ax.axvspan(
            selected_idx - 0.45, selected_idx + 0.45,
            alpha=0.08, color="green", label=f"Selected: {selected_experiment}"
        )

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Experiment Metrics", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "experiment_comparison.png"), dpi=300)
    plt.close()
