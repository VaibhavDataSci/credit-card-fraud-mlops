#!/usr/bin/env python3
"""CLI entry point for automated raw data validation."""

import json
import os
import sys
from pathlib import Path
import yaml

# Ensure src module is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.validation import DataValidator


def main():
    """Load configuration, execute validation, save report, and return exit status."""
    params_path = BASE_DIR / "params.yaml"

    if not params_path.exists():
        print(f"[ERROR] Configuration file not found at {params_path}")
        sys.exit(1)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    val_config = params.get("data_validation", {})
    # Ensure raw_data_path is resolved relative to BASE_DIR if needed
    raw_rel = val_config.get("raw_data_path", "data/raw/AIML DATASET.csv")
    raw_abs = BASE_DIR / raw_rel
    val_config["raw_data_path"] = str(raw_abs)

    print("==================================================")
    print("        RUNNING RAW DATASET VALIDATION            ")
    print("==================================================")
    print(f"Dataset Path : {raw_abs}")

    validator = DataValidator(val_config)
    report = validator.validate()

    # Create reports/validation output directory
    output_dir = BASE_DIR / "reports" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "data_validation_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    overall_status = report.get("overall_status", "FAILED")
    print(f"Validation Report Saved : {report_path}")
    print(f"Overall Validation Status: [{overall_status}]")
    print("--------------------------------------------------")

    checks = report.get("checks", {})
    if "dataset_size" in checks:
        size_info = checks["dataset_size"]
        print(f" - Dataset Dimensions  : {size_info['rows']} rows, {size_info['columns']} columns")

    if "schema" in checks:
        print(f" - Schema Validation   : {checks['schema']['status']}")

    if "target_column" in checks:
        target_info = checks["target_column"]
        print(f" - Target Column ({target_info['target_name']}) Status: {target_info['status']}")
        print(
            f" - Class Balance       : Non-Fraud={target_info['non_fraud_count']}, "
            f"Fraud={target_info['fraud_count']} ({target_info['fraud_percentage']:.3f}%)"
        )

    if "null_values" in checks:
        print(f" - Null Values Count   : {checks['null_values']['total_nulls']}")

    if "duplicates" in checks:
        print(f" - Duplicate Rows Count: {checks['duplicates']['duplicate_count']}")

    if "numerical_sanity" in checks:
        print(f" - Numerical Sanity    : {checks['numerical_sanity']['status']}")

    print("==================================================")

    if overall_status != "PASSED":
        print("[FAIL] Raw data validation failed. Review the report for details.")
        sys.exit(1)
    else:
        print("[SUCCESS] Raw data validation completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
