"""Unit tests for model comparison, threshold analysis, and selection logic."""

import json
import os

import numpy as np
import pytest

from src.models.compare import (
    ModelSelector,
    ThresholdAnalyzer,
    plot_experiment_comparison,
)


# ---------------------------------------------------------------
# ThresholdAnalyzer Tests
# ---------------------------------------------------------------

class TestThresholdAnalyzer:
    """Test suite for ThresholdAnalyzer class."""

    def test_analyze_returns_correct_structure(self):
        """Verify threshold analysis returns one result per threshold with required fields."""
        analyzer = ThresholdAnalyzer(thresholds=[0.3, 0.5, 0.7])
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.4, 0.8, 0.6, 0.9, 0.2, 0.55, 0.35])

        results = analyzer.analyze(y_true, y_proba)

        assert len(results) == 3
        required_keys = {
            "threshold", "precision", "recall", "f1_score",
            "true_positives", "true_negatives", "false_positives", "false_negatives",
        }
        for r in results:
            assert required_keys.issubset(r.keys())

    def test_analyze_threshold_affects_predictions(self):
        """Higher threshold should generally reduce recall and increase precision."""
        analyzer = ThresholdAnalyzer(thresholds=[0.3, 0.9])
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.4, 0.8, 0.6, 0.9, 0.2, 0.55, 0.35])

        results = analyzer.analyze(y_true, y_proba)

        low_thresh = results[0]   # 0.3
        high_thresh = results[1]  # 0.9

        # Higher threshold → fewer positives → recall should be lower or equal
        assert high_thresh["recall"] <= low_thresh["recall"]

    def test_analyze_perfect_predictions(self):
        """Perfect probabilities at threshold 0.5 should yield perfect metrics."""
        analyzer = ThresholdAnalyzer(thresholds=[0.5])
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.9, 1.0])

        results = analyzer.analyze(y_true, y_proba)

        assert results[0]["precision"] == 1.0
        assert results[0]["recall"] == 1.0
        assert results[0]["false_positives"] == 0
        assert results[0]["false_negatives"] == 0


# ---------------------------------------------------------------
# ModelSelector Tests
# ---------------------------------------------------------------

class TestModelSelector:
    """Test suite for ModelSelector class."""

    @pytest.fixture
    def mock_experiments(self):
        """Fixture providing mock experiment results for selection tests."""
        return {
            "experiment_a": {
                "precision": 0.05,
                "recall": 0.99,
                "f1_score": 0.09,
                "roc_auc": 0.999,
                "pr_auc": 0.95,
                "confusion_matrix": {
                    "true_negative": 100000,
                    "false_positive": 5000,
                    "false_negative": 2,
                    "true_positive": 198,
                },
            },
            "experiment_b": {
                "precision": 0.60,
                "recall": 0.96,
                "f1_score": 0.74,
                "roc_auc": 0.998,
                "pr_auc": 0.97,
                "confusion_matrix": {
                    "true_negative": 104800,
                    "false_positive": 200,
                    "false_negative": 8,
                    "true_positive": 192,
                },
            },
            "experiment_c": {
                "precision": 0.90,
                "recall": 0.80,
                "f1_score": 0.85,
                "roc_auc": 0.990,
                "pr_auc": 0.92,
                "confusion_matrix": {
                    "true_negative": 104980,
                    "false_positive": 20,
                    "false_negative": 40,
                    "true_positive": 160,
                },
            },
        }

    def test_select_best_picks_highest_pr_auc_with_sufficient_recall(self, mock_experiments):
        """Among experiments with recall >= 0.95, the one with highest PR-AUC should win."""
        selector = ModelSelector(min_recall=0.95)
        name, metrics = selector.select_best_experiment(mock_experiments)

        # experiment_b has recall=0.96 >= 0.95 AND highest PR-AUC=0.97
        assert name == "experiment_b"
        assert metrics["pr_auc"] == 0.97

    def test_select_penalizes_low_recall(self, mock_experiments):
        """Experiments below min_recall should be excluded unless no alternatives exist."""
        selector = ModelSelector(min_recall=0.95)
        name, _ = selector.select_best_experiment(mock_experiments)

        # experiment_c has recall=0.80 < 0.95, should NOT be selected
        assert name != "experiment_c"

    def test_select_falls_back_when_none_qualify(self):
        """When no experiment meets min_recall, all are considered with same criteria."""
        selector = ModelSelector(min_recall=0.99)
        experiments = {
            "low_recall_high_prauc": {
                "recall": 0.70, "pr_auc": 0.90, "f1_score": 0.80,
                "confusion_matrix": {"false_negative": 60},
            },
            "mid_recall_low_prauc": {
                "recall": 0.85, "pr_auc": 0.88, "f1_score": 0.82,
                "confusion_matrix": {"false_negative": 30},
            },
        }

        name, metrics = selector.select_best_experiment(experiments)
        # Neither meets 0.99. Same sort applies: highest PR-AUC wins → low_recall_high_prauc
        assert name == "low_recall_high_prauc"
        assert metrics["pr_auc"] == 0.90

    def test_select_best_threshold_qualifying(self):
        """Select threshold with highest F1 among those with recall >= min_recall."""
        selector = ModelSelector(min_recall=0.90)
        thresholds = [
            {"threshold": 0.5, "recall": 0.99, "f1_score": 0.10,
             "precision": 0.05, "false_positives": 5000, "false_negatives": 2},
            {"threshold": 0.7, "recall": 0.95, "f1_score": 0.60,
             "precision": 0.40, "false_positives": 300, "false_negatives": 10},
            {"threshold": 0.9, "recall": 0.80, "f1_score": 0.85,
             "precision": 0.90, "false_positives": 20, "false_negatives": 40},
        ]

        best = selector.select_best_threshold(thresholds)
        # threshold=0.7 has recall=0.95 >= 0.90 AND highest F1=0.60 among qualifying
        assert best["threshold"] == 0.7

    def test_select_best_threshold_fallback(self):
        """When no threshold meets min_recall, pick highest recall."""
        selector = ModelSelector(min_recall=0.99)
        thresholds = [
            {"threshold": 0.5, "recall": 0.90, "f1_score": 0.30,
             "precision": 0.20, "false_positives": 1000, "false_negatives": 20},
            {"threshold": 0.7, "recall": 0.85, "f1_score": 0.50,
             "precision": 0.35, "false_positives": 500, "false_negatives": 30},
        ]

        best = selector.select_best_threshold(thresholds)
        assert best["threshold"] == 0.5  # highest recall

    def test_generate_comparison_report_structure(self, mock_experiments, tmp_path):
        """Verify comparison report JSON has correct structure."""
        selector = ModelSelector()
        threshold_results = [
            {"threshold": 0.5, "precision": 0.05, "recall": 0.99, "f1_score": 0.09,
             "false_positives": 5000, "false_negatives": 2, "true_positives": 198, "true_negatives": 100000},
        ]
        selected_threshold = threshold_results[0]

        report = selector.generate_comparison_report(
            experiment_results=mock_experiments,
            selected_experiment="experiment_b",
            threshold_results=threshold_results,
            selected_threshold=selected_threshold,
            reports_dir=str(tmp_path),
        )

        assert "experiments" in report
        assert "selected_experiment" in report
        assert "threshold_analysis" in report
        assert "selected_threshold" in report
        assert report["selected_experiment"] == "experiment_b"
        assert len(report["experiments"]) == 3

        # Verify file was written
        report_file = tmp_path / "experiment_comparison.json"
        assert report_file.exists()
        with open(report_file) as f:
            saved = json.load(f)
        assert saved["selected_experiment"] == "experiment_b"

    def test_generate_metadata_structure(self, tmp_path):
        """Verify metadata JSON contains all required fields."""
        selector = ModelSelector()

        metadata = selector.generate_metadata(
            selected_experiment="test_exp",
            experiment_config={"use_smote": True, "use_scale_pos_weight": True,
                               "description": "Test strategy"},
            metrics={"precision": 0.5, "recall": 0.9, "f1_score": 0.6,
                     "roc_auc": 0.95, "pr_auc": 0.85},
            selected_threshold={"threshold": 0.7, "precision": 0.6, "recall": 0.88,
                                "f1_score": 0.72, "false_positives": 100, "false_negatives": 24},
            mlflow_run_id="test_run_123",
            mlflow_experiment_name="test_experiment",
            reports_dir=str(tmp_path),
        )

        required_fields = [
            "model_name", "model_type", "experiment_name", "mlflow_run_id",
            "selected_experiment", "imbalance_strategy", "use_smote",
            "use_scale_pos_weight", "selected_threshold", "evaluation_metrics",
            "selection_date",
        ]
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"

        assert metadata["model_name"] == "CreditCardFraudDetector"
        assert metadata["selected_threshold"] == 0.7
        assert "at_default_threshold" in metadata["evaluation_metrics"]
        assert "at_selected_threshold" in metadata["evaluation_metrics"]

        # Verify file was written
        metadata_file = tmp_path / "model_metadata.json"
        assert metadata_file.exists()


# ---------------------------------------------------------------
# Visualization Test
# ---------------------------------------------------------------

class TestVisualization:
    """Test suite for comparison visualization."""

    def test_plot_experiment_comparison_creates_file(self, tmp_path):
        """Verify plot is saved to the reports directory."""
        results = {
            "exp_a": {"precision": 0.5, "recall": 0.9, "f1_score": 0.6, "roc_auc": 0.95, "pr_auc": 0.85},
            "exp_b": {"precision": 0.7, "recall": 0.8, "f1_score": 0.75, "roc_auc": 0.93, "pr_auc": 0.88},
        }

        plot_experiment_comparison(results, selected_experiment="exp_b", reports_dir=str(tmp_path))

        plot_file = tmp_path / "experiment_comparison.png"
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0
