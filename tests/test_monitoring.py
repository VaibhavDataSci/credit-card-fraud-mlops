"""Tests for process-local API monitoring aggregates."""

import pytest

from app.monitoring import MonitoringState


@pytest.fixture
def monitor():
    return MonitoringState()


def test_initial_monitoring_state_is_empty(monitor):
    summary = monitor.summary()

    assert summary["requests"]["total"] == 0
    assert summary["requests"]["error_rate"] == 0.0
    assert summary["latency"]["count"] == 0
    assert summary["predictions"]["total"] == 0
    assert summary["data_quality"]["errors"] == 0


def test_request_and_latency_tracking(monitor):
    monitor.record_request("/predict", "POST", 200, 12.5)
    monitor.record_request("/health", "GET", 503, 4.0)

    summary = monitor.summary()
    assert summary["requests"]["total"] == 2
    assert summary["requests"]["errors"] == 1
    assert summary["requests"]["error_rate"] == pytest.approx(0.5)
    assert summary["requests"]["by_endpoint"]["POST /predict"] == 1
    assert summary["requests"]["by_status"]["503"] == 1
    assert summary["latency"]["count"] == 2
    assert summary["latency"]["average_ms"] == pytest.approx(8.25)
    assert summary["latency"]["min_ms"] == pytest.approx(4.0)
    assert summary["latency"]["max_ms"] == pytest.approx(12.5)


def test_prediction_distribution_and_threshold_tracking(monitor):
    monitor.record_prediction(0.95, True, 0.90)
    monitor.record_prediction(0.20, False, 0.90)

    predictions = monitor.summary()["predictions"]
    assert predictions["total"] == 2
    assert predictions["fraud"] == 1
    assert predictions["non_fraud"] == 1
    assert predictions["fraud_rate"] == pytest.approx(0.5)
    assert predictions["average_probability"] == pytest.approx(0.575)
    assert predictions["min_probability"] == pytest.approx(0.20)
    assert predictions["max_probability"] == pytest.approx(0.95)
    assert predictions["below_threshold"] == 1
    assert predictions["at_or_above_threshold"] == 1


def test_data_quality_errors_use_controlled_categories(monitor):
    monitor.record_data_quality_error("missing_field")
    monitor.record_data_quality_error("invalid_amount")
    monitor.record_data_quality_error("untrusted-user-value")

    quality = monitor.summary()["data_quality"]
    assert quality["errors"] == 3
    assert quality["by_category"] == {
        "missing_field": 1,
        "invalid_amount": 1,
        "invalid_type": 1,
    }


def test_reset_returns_monitoring_to_initial_state(monitor):
    monitor.record_request("/predict", "POST", 200, 2.0)
    monitor.record_prediction(0.91, True, 0.90)
    monitor.record_data_quality_error("invalid_balance")

    monitor.reset()

    assert monitor.summary()["requests"]["total"] == 0
    assert monitor.summary()["predictions"]["total"] == 0
    assert monitor.summary()["data_quality"]["errors"] == 0
