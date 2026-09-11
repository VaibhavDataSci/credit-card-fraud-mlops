#!/usr/bin/env python3
"""Run bounded Evidently drift detection on reference and current data."""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.monitoring.drift import run_drift_detection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/processed/cleaned.parquet")
    parser.add_argument("--current", required=True)
    parser.add_argument("--html", default="reports/drift/drift_report.html")
    parser.add_argument("--json", dest="json_path", default="reports/drift/drift_summary.json")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--drift-share-threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = run_drift_detection(
        reference_path=args.reference,
        current_path=args.current,
        html_path=args.html,
        json_path=args.json_path,
        sample_size=args.sample_size,
        drift_share_threshold=args.drift_share_threshold,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "feature_results"}, indent=2))
    if result["investigation_required"]:
        print("Investigation recommended: distributional drift was detected.")
    else:
        print("No feature drift detected at the configured methodology threshold.")


if __name__ == "__main__":
    main()
