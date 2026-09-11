"""Tests for the explicit human approval gate."""

import json
from types import SimpleNamespace

import pytest

from src.models.approval import decide_candidate


class FakeClient:
    def __init__(self, tracking_uri):
        self.champion = SimpleNamespace(version="1", run_id="champion-run")
        self.candidate = SimpleNamespace(version="2", run_id="candidate-run")
        self.alias_updates = []

    def get_model_version_by_alias(self, name, alias):
        return self.champion

    def get_model_version(self, name, version):
        assert version == "2"
        return self.candidate

    def set_registered_model_alias(self, name, alias, version):
        self.alias_updates.append((name, alias, version))
        self.champion = self.candidate


def pending_report(path):
    report = {
        "recommended_decision": "PROMOTE",
        "approval_state": "PENDING_APPROVAL",
        "candidate": {"run_id": "candidate-run", "model_version": "2", "metrics": {"f1_score": 0.9, "recall": 0.9, "pr_auc": 0.9, "precision": 0.9, "roc_auc": 0.9}},
        "champion": {"run_id": "champion-run", "model_version": "1", "metrics": {"f1_score": 0.8, "recall": 0.8, "pr_auc": 0.8, "precision": 0.8, "roc_auc": 0.8}},
        "comparison": {"decision": "PROMOTE", "checks": {"f1_score": True}},
    }
    path.write_text(json.dumps(report))


def test_candidate_starts_pending_approval(tmp_path):
    path = tmp_path / "report.json"
    pending_report(path)
    assert json.loads(path.read_text())["approval_state"] == "PENDING_APPROVAL"


def test_reject_keeps_champion_and_records_reason(monkeypatch, tmp_path):
    fake = FakeClient("file:test")
    monkeypatch.setattr("src.models.approval.MlflowClient", lambda tracking_uri: fake)
    report = tmp_path / "report.json"
    pending_report(report)

    result = decide_candidate(report, "file:test", "CreditCardFraudDetector", "REJECT", "operator", "Recall insufficient", tmp_path / "history.json", tmp_path / "approval.md")

    assert result["champion_after"] == "1"
    assert fake.alias_updates == []
    history = json.loads((tmp_path / "history.json").read_text())
    assert history[0]["decision"] == "REJECTED"
    assert history[0]["reason"] == "Recall insufficient"


def test_approve_requires_exact_keyword_and_promotes(monkeypatch, tmp_path):
    fake = FakeClient("file:test")
    monkeypatch.setattr("src.models.approval.MlflowClient", lambda tracking_uri: fake)
    report = tmp_path / "report.json"
    pending_report(report)

    with pytest.raises(ValueError, match="APPROVE or REJECT"):
        decide_candidate(report, "file:test", "CreditCardFraudDetector", "yes", "operator", approval_history_path=tmp_path / "history.json")

    result = decide_candidate(report, "file:test", "CreditCardFraudDetector", "APPROVE", "operator", approval_history_path=tmp_path / "history.json", approval_report_path=tmp_path / "approval.md")
    assert result["approval"]["decision"] == "APPROVED"
    assert result["approval"]["promotion_result"] == "PROMOTED"
    assert result["champion_after"] == "2"
    assert fake.alias_updates == [("CreditCardFraudDetector", "champion", "2")]
    assert "APPROVED" in (tmp_path / "approval.md").read_text()


def test_stale_candidate_is_rejected(monkeypatch, tmp_path):
    fake = FakeClient("file:test")
    fake.champion = SimpleNamespace(version="3", run_id="new-champion")
    monkeypatch.setattr("src.models.approval.MlflowClient", lambda tracking_uri: fake)
    report = tmp_path / "report.json"
    pending_report(report)

    with pytest.raises(ValueError, match="Stale approval"):
        decide_candidate(report, "file:test", "CreditCardFraudDetector", "APPROVE", "operator")


def test_failed_criteria_cannot_be_approved(monkeypatch, tmp_path):
    fake = FakeClient("file:test")
    monkeypatch.setattr("src.models.approval.MlflowClient", lambda tracking_uri: fake)
    report = tmp_path / "report.json"
    pending_report(report)
    data = json.loads(report.read_text())
    data["comparison"]["decision"] = "REJECT"
    report.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="criteria"):
        decide_candidate(report, "file:test", "CreditCardFraudDetector", "APPROVE", "operator")
    assert fake.alias_updates == []
