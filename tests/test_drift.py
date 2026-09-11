"""Lightweight deterministic tests for Evidently drift detection."""

import json

import pandas as pd
import pytest

from src.monitoring.drift import DriftDataError, DriftDetector


FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
    "amount_to_balance_ratio",
]


def make_data(shifted: bool = False, rows: int = 100) -> pd.DataFrame:
    if shifted:
        types = ["TRANSFER"] * rows
        amount = [10000.0 + index for index in range(rows)]
    else:
        types = (["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"] * rows)[:rows]
        amount = [float(100 + index) for index in range(rows)]
    old_origin = [1000.0 + index for index in range(rows)]
    new_origin = [old - 100.0 for old in old_origin]
    old_destination = [50.0 + index for index in range(rows)]
    new_destination = [old + 100.0 for old in old_destination]
    return pd.DataFrame(
        {
            "type": types,
            "amount": amount,
            "oldbalanceOrg": old_origin,
            "newbalanceOrig": new_origin,
            "oldbalanceDest": old_destination,
            "newbalanceDest": new_destination,
            "balance_diff_orig": [100.0] * rows,
            "balance_diff_dest": [100.0] * rows,
            "amount_to_balance_ratio": [value / (old + 1) for value, old in zip(amount, old_origin)],
        }
    )


def test_reference_and_current_schema_are_compatible():
    detector = DriftDetector()
    detector.validate_data(make_data(), "reference")
    detector.validate_data(make_data(), "current")


def test_missing_feature_is_rejected():
    data = make_data().drop(columns=["amount"])

    with pytest.raises(DriftDataError, match="amount"):
        DriftDetector().validate_data(data, "current")


def test_numeric_and_categorical_drift_is_detected():
    result = DriftDetector(drift_share_threshold=0.2).compare(make_data(), make_data(shifted=True))

    assert result["drifted_features"]
    assert "amount" in result["drifted_features"]
    assert "type" in result["drifted_features"]
    assert result["investigation_required"] is True
    assert result["drift_share"] > 0


def test_identical_data_has_no_drift():
    data = make_data()
    result = DriftDetector().compare(data, data.copy())

    assert result["drifted_features"] == []
    assert result["drift_share"] == 0.0
    assert result["dataset_drift_detected"] is False
    assert result["investigation_required"] is False


def test_invalid_non_finite_numeric_data_is_rejected():
    data = make_data()
    data.loc[0, "amount"] = float("inf")

    with pytest.raises(DriftDataError, match="non-finite"):
        DriftDetector().validate_data(data, "current")


def test_empty_data_is_rejected():
    with pytest.raises(DriftDataError, match="empty"):
        DriftDetector().validate_data(make_data().iloc[0:0], "current")


def test_reports_are_saved_as_html_and_json(tmp_path):
    detector = DriftDetector()
    result = detector.compare(make_data(), make_data(shifted=True))
    html_path = tmp_path / "drift_report.html"
    json_path = tmp_path / "drift_summary.json"

    detector.save_reports(result, html_path, json_path)

    assert html_path.exists()
    assert "drift" in html_path.read_text(encoding="utf-8").lower()
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["features_analyzed"] == FEATURES
    assert summary["investigation_required"] is True
    assert "evidently_report" not in summary
