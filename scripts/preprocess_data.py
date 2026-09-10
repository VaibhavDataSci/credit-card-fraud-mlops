#!/usr/bin/env python3
"""CLI entry point for data preprocessing and feature engineering."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

# Ensure src module is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.preprocessing import DataPreprocessor


def main():
    """Execute preprocessing pipeline, save Parquet dataset, and output report."""
    params_path = BASE_DIR / "params.yaml"

    if not params_path.exists():
        print(f"[ERROR] Configuration file not found at {params_path}")
        sys.exit(1)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    prep_config = params.get("preprocessing", {})
    raw_rel = prep_config.get("raw_data_path", "data/raw/AIML DATASET.csv")
    preproc_rel = prep_config.get("preprocessed_data_path", "data/processed/preprocessed.parquet")

    raw_abs = BASE_DIR / raw_rel
    preproc_abs = BASE_DIR / preproc_rel

    prep_config["raw_data_path"] = str(raw_abs)
    prep_config["preprocessed_data_path"] = str(preproc_abs)

    print("==================================================")
    print("        RUNNING RAW DATA PREPROCESSING STAGE      ")
    print("==================================================")
    print(f"Raw Input Dataset : {raw_abs}")
    print(f"Preprocessed Output: {preproc_abs}")

    try:
        preprocessor = DataPreprocessor(prep_config)
        processed_df, metadata = preprocessor.preprocess_raw()
        saved_path = preprocessor.save_processed_data(processed_df, output_path=str(preproc_abs))

        # Generate processing report
        output_dir = BASE_DIR / "reports" / "preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "preprocessing_report.json"

        file_size_bytes = os.path.getsize(saved_path)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_dataset_path": str(raw_abs),
            "preprocessed_dataset_path": str(saved_path),
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "input_shape": metadata["input_shape"],
            "output_shape": metadata["output_shape"],
            "features_removed": metadata["features_removed"],
            "final_columns": metadata["final_columns"],
            "target_column": metadata["target_column"],
            "status": "SUCCESS",
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("--------------------------------------------------")
        print(f"Preprocessed Dataset Saved: {saved_path} ({report['file_size_mb']} MB)")
        print(f"Processing Report Saved   : {report_path}")
        print(f" - Input Shape            : {metadata['input_shape']['rows']} rows, {metadata['input_shape']['columns']} cols")
        print(f" - Output Shape           : {metadata['output_shape']['rows']} rows, {metadata['output_shape']['columns']} cols")
        print(f" - Features Removed       : {metadata['features_removed']}")
        print("==================================================")
        print("[SUCCESS] Data preprocessing completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"[FAIL] Data preprocessing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
