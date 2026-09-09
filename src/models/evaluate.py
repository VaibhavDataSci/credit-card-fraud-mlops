"""Model evaluation module for Credit Card Fraud Detection pipeline."""

import json
import os
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Non-interactive backend for headless environments
plt.switch_backend("Agg")


class ModelEvaluator:
    """Evaluator class for generating fraud detection model metrics and visual plots."""

    def __init__(self, reports_dir: str = "reports/model"):
        """Initialize evaluator with output reports directory.

        Args:
            reports_dir: Output directory path for evaluation artifacts.
        """
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def evaluate(
        self, pipeline: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Compute evaluation metrics on untouched test set.

        Args:
            pipeline: Trained model or pipeline object.
            X_test: Test features DataFrame.
            y_test: Test target Series.

        Returns:
            Dictionary containing computed evaluation metrics and classification details.
        """
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]

        class_report_dict = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "test_samples": int(len(y_test)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
        }

        # 1. Save JSON reports
        with open(os.path.join(self.reports_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(self.reports_dir, "classification_report.json"), "w") as f:
            json.dump(class_report_dict, f, indent=2)

        # 2. Plot & Save Confusion Matrix
        self.plot_confusion_matrix(cm)

        # 3. Plot & Save ROC Curve
        self.plot_roc_curve(y_test, y_proba, roc_auc)

        # 4. Plot & Save Precision-Recall Curve
        self.plot_precision_recall_curve(y_test, y_proba, pr_auc)

        return metrics

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix heatmap.

        Args:
            cm: 2x2 confusion matrix array.
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Fraud (0)", "Fraud (1)"],
            yticklabels=["Non-Fraud (0)", "Fraud (1)"],
        )
        plt.title("Test Set Confusion Matrix", fontsize=12, fontweight="bold")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

    def plot_roc_curve(self, y_test: pd.Series, y_proba: np.ndarray, roc_auc: float):
        """Plot and save ROC Curve.

        Args:
            y_test: True test labels.
            y_proba: Predicted fraud probabilities.
            roc_auc: ROC-AUC score value.
        """
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "roc_curve.png"), dpi=300)
        plt.close()

    def plot_precision_recall_curve(
        self, y_test: pd.Series, y_proba: np.ndarray, pr_auc: float
    ):
        """Plot and save Precision-Recall Curve.

        Args:
            y_test: True test labels.
            y_proba: Predicted fraud probabilities.
            pr_auc: PR-AUC (Average Precision) score value.
        """
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, color="#e74c3c", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "precision_recall_curve.png"), dpi=300)
        plt.close()
