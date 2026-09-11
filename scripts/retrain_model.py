#!/usr/bin/env python3
"""Validate new labeled data, train an MLflow candidate, compare, and optionally promote."""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.retraining import run_retraining


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-data", required=True)
    parser.add_argument("--params", default=str(BASE_DIR / "params.yaml"))
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--report-dir", default=str(BASE_DIR / "reports/model"))
    parser.add_argument("--promote", action="store_true", help="Promote only if every gate passes")
    args = parser.parse_args()
    report = run_retraining(args.new_data, args.params, args.tracking_uri, args.promote, args.report_dir)
    print(json.dumps({key: value for key, value in report.items() if key not in {"candidate", "champion"}}, indent=2))


if __name__ == "__main__":
    main()