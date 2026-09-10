"""Unit tests for DVC pipeline configuration and stage definitions."""

from pathlib import Path
import pytest
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def test_dvc_yaml_exists():
    """Test that dvc.yaml file exists in workspace root."""
    dvc_path = BASE_DIR / "dvc.yaml"
    assert dvc_path.exists(), "dvc.yaml does not exist in workspace root!"


def test_dvc_stages_configuration():
    """Test that dvc.yaml contains all required stages in order."""
    dvc_path = BASE_DIR / "dvc.yaml"
    with open(dvc_path, "r") as f:
        dvc_config = yaml.safe_load(f)

    stages = dvc_config.get("stages", {})
    required_stages = ["validate", "preprocess", "features", "train", "evaluate"]

    for stage_name in required_stages:
        assert stage_name in stages, f"Required stage '{stage_name}' missing from dvc.yaml"

    # Verify validate stage
    val_stage = stages["validate"]
    assert "cmd" in val_stage
    assert "scripts/validate_data.py" in val_stage["cmd"]

    # Verify preprocess stage
    prep_stage = stages["preprocess"]
    assert "cmd" in prep_stage
    assert "scripts/preprocess_data.py" in prep_stage["cmd"]
    assert "data/processed/preprocessed.parquet" in prep_stage["outs"]

    # Verify features stage
    feat_stage = stages["features"]
    assert "cmd" in feat_stage
    assert "scripts/create_features.py" in feat_stage["cmd"]
    assert "data/processed/cleaned.parquet" in feat_stage["outs"]

    # Verify train stage
    train_stage = stages["train"]
    assert "cmd" in train_stage
    assert "scripts/train_model.py" in train_stage["cmd"]
    assert "models/xgboost_fraud_model.joblib" in train_stage["outs"]

    # Verify evaluate stage
    eval_stage = stages["evaluate"]
    assert "cmd" in eval_stage
    assert "scripts/evaluate_and_register.py" in eval_stage["cmd"]
    assert "reports/model/metrics.json" in eval_stage["outs"]
