"""Process-local aggregate monitoring for the FastAPI inference service."""

from collections import Counter
from threading import Lock
from typing import Any


class MonitoringState:
    """Thread-safe, bounded aggregate metrics for one API process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", Lock()):
            self.total_requests = 0
            self.error_requests = 0
            self.request_counts: Counter[str] = Counter()
            self.status_counts: Counter[str] = Counter()
            self.latencies_ms: list[float] = []
            self.predicted_fraud_count = 0
            self.predicted_non_fraud_count = 0
            self.probability_count = 0
            self.probability_sum = 0.0
            self.probability_min: float | None = None
            self.probability_max: float | None = None
            self.below_threshold_count = 0
            self.at_or_above_threshold_count = 0
            self.data_quality_errors = 0
            self.data_quality_by_category: Counter[str] = Counter()

    def record_request(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self.request_counts[f"{method} {path}"] += 1
            self.status_counts[str(status_code)] += 1
            if status_code >= 400:
                self.error_requests += 1
            self.latencies_ms.append(float(latency_ms))

    def record_prediction(self, probability: float, is_fraud: bool, threshold: float) -> None:
        with self._lock:
            if is_fraud:
                self.predicted_fraud_count += 1
                self.at_or_above_threshold_count += 1
            else:
                self.predicted_non_fraud_count += 1
                self.below_threshold_count += 1
            self.probability_count += 1
            self.probability_sum += probability
            self.probability_min = probability if self.probability_min is None else min(self.probability_min, probability)
            self.probability_max = probability if self.probability_max is None else max(self.probability_max, probability)

    def record_data_quality_error(self, category: str) -> None:
        allowed_categories = {"missing_field", "invalid_type", "invalid_amount", "invalid_balance", "invalid_numeric_value"}
        category = category if category in allowed_categories else "invalid_type"
        with self._lock:
            self.data_quality_errors += 1
            self.data_quality_by_category[category] += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total_predictions = self.predicted_fraud_count + self.predicted_non_fraud_count
            latency_count = len(self.latencies_ms)
            return {
                "requests": {
                    "total": self.total_requests,
                    "errors": self.error_requests,
                    "error_rate": self.error_requests / self.total_requests if self.total_requests else 0.0,
                    "by_endpoint": dict(self.request_counts),
                    "by_status": dict(self.status_counts),
                },
                "latency": {
                    "count": latency_count,
                    "average_ms": sum(self.latencies_ms) / latency_count if latency_count else 0.0,
                    "min_ms": min(self.latencies_ms) if latency_count else 0.0,
                    "max_ms": max(self.latencies_ms) if latency_count else 0.0,
                },
                "predictions": {
                    "total": total_predictions,
                    "fraud": self.predicted_fraud_count,
                    "non_fraud": self.predicted_non_fraud_count,
                    "fraud_rate": self.predicted_fraud_count / total_predictions if total_predictions else 0.0,
                    "probability_count": self.probability_count,
                    "average_probability": self.probability_sum / self.probability_count if self.probability_count else 0.0,
                    "min_probability": self.probability_min,
                    "max_probability": self.probability_max,
                    "below_threshold": self.below_threshold_count,
                    "at_or_above_threshold": self.at_or_above_threshold_count,
                },
                "data_quality": {
                    "errors": self.data_quality_errors,
                    "by_category": dict(self.data_quality_by_category),
                },
            }
