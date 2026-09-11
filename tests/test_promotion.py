"""Unit tests for safety-gated candidate promotion decisions."""

import json

import pytest

from src.models.promotion import evaluate_promotion, write_promotion_reports


def metrics(pr_auc=0.8, recall=0.9, f1=0.85, false_negative=10):
    return {
        "precision": 0.8,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": 0.95,
        "pr_auc": pr_auc,
        "confusion_matrix": {"true_negative": 90, "false_positive": 10, "false_negative": false_negative, "true_positive": 90},
    }


def test_better_candidate_is_promoted():
    decision = evaluate_promotion(
        {"min_recall_delta": 0.0, "min_pr_auc_delta": 0.0, "min_f1_delta": 0.0},
        metrics(pr_auc=0.85, recall=0.92, f1=0.88, false_negative=8),
        metrics(),
    )

    assert decision["decision"] == "PROMOTE"
    assert decision["failed_criteria"] == []


def test_worse_candidate_is_rejected():
    decision = evaluate_promotion({}, metrics(pr_auc=0.7, recall=0.8, f1=0.75, false_negative=20), metrics())

    assert decision["decision"] == "REJECT"
    assert "recall" in decision["failed_criteria"]
    assert "pr_auc" in decision["failed_criteria"]
    assert "false_negatives" in decision["failed_criteria"]


def test_recall_regression_is_rejected_even_with_f1_improvement():
    decision = evaluate_promotion({}, metrics(pr_auc=0.85, recall=0.89, f1=0.90, false_negative=11), metrics())

    assert decision["decision"] == "REJECT"
    assert "recall" in decision["failed_criteria"]


def test_pr_auc_regression_is_rejected():
    decision = evaluate_promotion({}, metrics(pr_auc=0.79, recall=0.91, f1=0.86, false_negative=9), metrics())

    assert decision["decision"] == "REJECT"
    assert "pr_auc" in decision["failed_criteria"]


def test_reports_contain_actual_decision_and_metrics(tmp_path):
    comparison = evaluate_promotion({}, metrics(pr_auc=0.85, recall=0.92, f1=0.88, false_negative=8), metrics())
    report = {
        "decision": comparison["decision"],
        "candidate": {"run_id": "candidate-run", "model_version": "2", "metrics": metrics(pr_auc=0.85, recall=0.92, f1=0.88, false_negative=8)},
        "champion": {"run_id": "champion-run", "model_version": "1", "metrics": metrics()},
        "comparison": comparison,
        "criteria": {},
        "failed_criteria": comparison["failed_criteria"],
        "reason": comparison["reason"],
    }
    write_promotion_reports(report, tmp_path / "decision.json", tmp_path / "decision.md")

    saved = json.loads((tmp_path / "decision.json").read_text())
    assert saved["decision"] == "PROMOTE"
    assert "PROMOTE" in (tmp_path / "decision.md").read_text()


def test_promotion_requires_explicit_promote_decision(monkeypatch):
    from src.models.promotion import promote_candidate

    with pytest.raises(ValueError, match="PROMOTE"):
        promote_candidate("file:/tmp/test", "CreditCardFraudDetector", "2", {"decision": "REJECT"})
