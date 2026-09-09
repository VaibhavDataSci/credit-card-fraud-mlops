# Credit Card Fraud Detection — MLOps Pipeline

## Project Overview

This repository contains an end-to-end, production-ready **Credit Card Fraud Detection MLOps pipeline** built with XGBoost.

The objective of this project is to implement a robust, reproducible, deployable, and continuously monitored machine learning system for financial fraud detection.

The full MLOps workflow will incorporate:
* **Data Versioning**: DVC dataset tracking and remote storage reproducibility.
* **Data Validation & Quality**: Automated schema, data type, nulls, duplicates, and numerical sanity checks (`src/data/validation.py`).
* **Exploratory Data Analysis**: Data distribution, class imbalance, and pattern visualizations (`reports/eda/`).
* **Preprocessing & Feature Engineering**: Reusable transformers for consistent training and inference.
* **Imbalance Handling**: SMOTE oversampling strictly applied to training data.
* **Model Training**: Hyperparameter-tuned XGBoost classifier.
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
Current Development Phase: Phase 3 — Data Validation + EDA
```

Phase 1 established the repository foundation and environment configuration.
Phase 2 initialized **DVC** dataset versioning and remote storage.
Phase 3 implements **Automated Data Validation** (`src/data/validation.py`, `scripts/validate_data.py`), unit testing suite (`tests/test_data_validation.py`), structured reporting (`reports/validation/data_validation_report.json`), and comprehensive **Exploratory Data Analysis** (`scripts/run_eda.py`, `reports/eda/`, `notebooks/phase3_data_validation_eda.ipynb`).

---

## Dataset & Versioning

### Dataset Details
* **Filename**: `AIML DATASET.csv`
* **Location**: `data/raw/AIML DATASET.csv`
* **Format**: CSV (`6,362,620` rows × `11` columns, `~470.67 MB`)
* **Domain & Purpose**: Synthetic financial transaction log (PaySim) for detecting fraudulent mobile money transactions (`isFraud`).
* **Columns**: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.

### Phase 3 Execution Commands

```bash
# Run automated data validation script
python scripts/validate_data.py

# Run validation unit tests
pytest tests/test_data_validation.py

# Generate EDA visualizations and plots
python scripts/run_eda.py
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
│   └── processed/          # Cleaned & transformed datasets
│
├── notebooks/              # Exploratory notebooks
│   ├── xgboost_experiments.ipynb
│   └── phase3_data_validation_eda.ipynb
│
├── src/                    # Core Python package
│   ├── __init__.py
│   ├── config.py           # Centralized environment variable configuration
│   ├── data/
│   │   ├── __init__.py
│   │   └── validation.py   # Automated data validation module
│   ├── features/           # Preprocessing & feature engineering modules
│   ├── models/             # Model training, SMOTE & evaluation modules
│   └── monitoring/         # Data & model drift detection modules
│
├── app/                    # FastAPI application & API endpoints
├── tests/                  # Pytest unit & integration tests
│   └── test_data_validation.py
│
├── configs/                # Environment & deployment configurations
├── reports/                # Validation reports & EDA visualizations
│   ├── validation/
│   │   └── data_validation_report.json
│   └── eda/
│       ├── class_distribution.png
│       ├── transaction_type_distribution.png
│       ├── fraud_by_transaction_type.png
│       ├── transaction_amount_distribution.png
│       ├── fraud_amount_comparison.png
│       ├── balance_analysis.png
│       ├── correlation_matrix.png
│       └── eda_findings.md
│
├── scripts/                # Execution scripts
│   ├── validate_data.py
│   └── run_eda.py
│
├── ProjectDetails.md       # Master project specification
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore patterns
└── params.yaml             # Pipeline and validation configurations
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

### 3. Run Data Validation & EDA
```bash
python scripts/validate_data.py
pytest tests/test_data_validation.py
python scripts/run_eda.py
```
