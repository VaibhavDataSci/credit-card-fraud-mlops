# Credit Card Fraud Detection — MLOps Pipeline

## Project Overview

This repository contains an end-to-end, production-ready **Credit Card Fraud Detection MLOps pipeline** built with XGBoost.

The objective of this project is to implement a robust, reproducible, deployable, and continuously monitored machine learning system for financial fraud detection.

The full MLOps workflow will incorporate:
* **Data Versioning**: DVC dataset tracking and remote storage reproducibility.
* **Data Validation & Quality**: Automated schema, data type, nulls, duplicates, and numerical sanity checks (`src/data/validation.py`).
* **Exploratory Data Analysis**: Data distribution, class imbalance, and pattern visualizations (`reports/eda/`).
* **Preprocessing & Feature Engineering**: Leak-free, vectorized feature engineering and column transformations (`src/features/feature_engineering.py`, `src/data/preprocessing.py`).
* **Model Training**: Baseline XGBoost classifier training with `ImbPipeline` (scaling + One-Hot Encoding + training-only SMOTE oversampling).
* **Model Evaluation**: Comprehensive metrics evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC) and diagnostic plots (`reports/model/`).
* **Data & Pipeline Versioning**: DVC pipeline (`dvc.yaml`) for dataset lineage and execution reproducibility.
* **Experiment Tracking & Registry**: MLflow for metrics, parameters, artifacts logging, and model lifecycle management.
* **API Serving**: FastAPI REST endpoints for model inference and health monitoring.
* **Containerization**: Docker & Docker Compose for isolated application deployment.
* **Automated Testing**: Unit and integration tests powered by Pytest.
* **CI/CD Automation**: GitHub Actions for automated integration, testing, and container build.
* **Monitoring & Drift Detection**: Real-time performance tracking and data drift detection using Evidently.
* **Automated Retraining**: Data-driven retraining loop and champion vs. candidate model comparison.

---

## Current Status

```text
Current Development Phase: Phase 5 — Model Training + Evaluation
```

Phase 1 established the repository foundation and environment configuration.
Phase 2 initialized **DVC** dataset versioning and remote storage.
Phase 3 implemented **Automated Data Validation** and **Exploratory Data Analysis**.
Phase 4 implemented **Preprocessing & Feature Engineering** (`data/processed/cleaned.parquet`).
Phase 5 implements **Model Training & Evaluation** (`src/models/train.py`, `src/models/evaluate.py`, `scripts/train_model.py`), generating trained model artifact `models/xgboost_fraud_model.joblib` and test evaluation reports/plots (`reports/model/`).

---

## Dataset & Versioning

### Dataset Details
* **Raw Dataset**: `AIML DATASET.csv` (`data/raw/AIML DATASET.csv`, `6,362,620` rows × `11` columns)
* **Processed Dataset**: `cleaned.parquet` (`data/processed/cleaned.parquet`, `6,362,620` rows × `10` columns, `249.57 MB`)
* **Trained Model Artifact**: `models/xgboost_fraud_model.joblib` (`1.88 MB`)

### Pipeline Execution Commands

```bash
# 1. Run automated data validation
python scripts/validate_data.py

# 2. Run data preprocessing & feature engineering
python scripts/preprocess_data.py

# 3. Train XGBoost model & evaluate test set
python scripts/train_model.py

# 4. Run unit test suite
pytest tests/
```

---

## Planned Architecture

```text
Raw Dataset
    ↓
DVC
    ↓
Data Validation
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
SMOTE (training data only)
    ↓
XGBoost
    ↓
MLflow Experiment Tracking
    ↓
Model Registry
    ↓
Model Validation
    ↓
FastAPI
    ↓
Docker
    ↓
CI/CD
    ↓
Monitoring
    ↓
Drift Detection
    ↓
Retraining
    ↓
New Model Version
    ↓
Champion Model
```

---

## Repository Structure

```text
credit-card-fraud-mlops/
│
├── .dvc/                   # DVC configuration & local storage remote
│
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
│
├── data/
│   ├── raw/                # Immutable raw datasets (tracked by DVC)
│   └── processed/          # Cleaned & transformed datasets (cleaned.parquet)
│
├── models/                 # Model binary artifacts (xgboost_fraud_model.joblib)
│
├── notebooks/              # Exploratory notebooks
│   ├── xgboost_experiments.ipynb
│   └── phase3_data_validation_eda.ipynb
│
├── src/                    # Core Python package
│   ├── __init__.py
│   ├── config.py           # Centralized environment variable configuration
│   ├── data/
│   │   ├── validation.py   # Automated data validation module
│   │   └── preprocessing.py# Data preprocessing module
│   ├── features/
│   │   └── feature_engineering.py # Feature engineering module
│   ├── models/
│   │   ├── train.py        # Model training module
│   │   └── evaluate.py     # Model evaluation module
│   └── monitoring/         # Data & model drift detection modules
│
├── app/                    # FastAPI application & API endpoints
├── tests/                  # Pytest unit & integration tests
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   ├── test_model_training.py
│   └── test_model_evaluation.py
│
├── configs/                # Environment & deployment configurations
├── reports/                # Validation reports, preprocessing reports & model evaluation
│   ├── validation/
│   ├── preprocessing/
│   ├── eda/
│   └── model/
│       ├── metrics.json
│       ├── classification_report.json
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── precision_recall_curve.png
│
├── scripts/                # Execution scripts
│   ├── validate_data.py
│   ├── run_eda.py
│   ├── preprocess_data.py
│   └── train_model.py
│
├── ProjectDetails.md       # Master project specification
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore patterns
└── params.yaml             # Pipeline and model configurations
```

---

## Getting Started

### 1. Clone the repository & Install Dependencies
```bash
git clone <repository-url>
cd credit-card-fraud-mlops
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch Dataset via DVC
```bash
dvc pull
```

### 3. Run Pipeline Stages & Tests
```bash
python scripts/validate_data.py
python scripts/preprocess_data.py
python scripts/train_model.py
pytest tests/
```
