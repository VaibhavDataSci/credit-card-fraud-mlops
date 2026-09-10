#!/usr/bin/env python3
"""CLI entry point for feature engineering stage."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml

# Ensure src module is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.features.feature_engineering import FeatureEngineer


def main():
    """Execute feature engineering stage and save cleaned parquet file."""
    params_path = BASE_DIR / "params.yaml"

    if not params_path.exists():
        print(f"[ERROR] Configuration file not found at {params_path}")
        sys.exit(1)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    prep_config = params.get("preprocessing", {})
    fe_config = params.get("feature_engineering", {})

    preprocessed_rel = prep_config.get("preprocessed_data_path", "data/processed/preprocessed.parquet")
    cleaned_rel = prep_config.get("processed_data_path", "data/processed/cleaned.parquet")

    preprocessed_abs = BASE_DIR / preprocessed_rel
    cleaned_abs = BASE_DIR / cleaned_rel

    if not preprocessed_abs.exists():
        print(f"[ERROR] Preprocessed dataset not found at {preprocessed_abs}")
        sys.exit(1)

    print("==================================================")
    print("      RUNNING FEATURE ENGINEERING PIPELINE        ")
    print("==================================================")
    print(f"Preprocessed Input : {preprocessed_abs}")
    print(f"Cleaned Output     : {cleaned_abs}")

    try:
        df = pd.read_parquet(preprocessed_abs)
        input_shape = df.shape
        input_cols = list(df.columns)

        engineer = FeatureEngineer(fe_config)
        cleaned_df = engineer.transform(df)

        output_shape = cleaned_df.shape
        created_features = [c for c in cleaned_df.columns if c not in input_cols]

        cleaned_abs.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_parquet(cleaned_abs, index=False)

        # Save feature report
        output_dir = BASE_DIR / "reports" / "preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "feature_engineering_report.json"

        file_size_bytes = os.path.getsize(cleaned_abs)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_path": str(preprocessed_abs),
            "output_path": str(cleaned_abs),
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "input_shape": {"rows": int(input_shape[0]), "columns": int(input_shape[1])},
            "output_shape": {"rows": int(output_shape[0]), "columns": int(output_shape[1])},
            "features_created": created_features,
            "status": "SUCCESS",
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("--------------------------------------------------")
        print(f"Cleaned Dataset Saved  : {cleaned_abs} ({report['file_size_mb']} MB)")
        print(f"Feature Report Saved   : {report_path}")
        print(f" - Input Shape         : {input_shape[0]} rows, {input_shape[1]} cols")
        print(f" - Output Shape        : {output_shape[0]} rows, {output_shape[1]} cols")
        print(f" - Features Created    : {created_features}")
        print("==================================================")
        print("[SUCCESS] Feature engineering completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"[FAIL] Feature engineering failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
