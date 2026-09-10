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
Current Development Phase: Phase 8 — MLflow Model Registry & Reproducible DVC Pipeline
```

Phase 1 established the repository foundation and environment configuration.
Phase 2 initialized **DVC** dataset versioning and remote storage.
Phase 3 implemented **Automated Data Validation** and **Exploratory Data Analysis**.
Phase 4 implemented **Preprocessing & Feature Engineering** (`data/processed/cleaned.parquet`).
Phase 5 implemented **Model Training & Evaluation** (`src/models/train.py`, `src/models/evaluate.py`, `scripts/train_model.py`).
Phase 6 integrated **MLflow Experiment Tracking** (`src/models/tracking.py`).
Phase 7 implemented **Model Comparison & Selection** — selected `scale_weight_only` at operating threshold `0.90` as deployment candidate.
Phase 8 implemented **MLflow Model Registry** (`CreditCardFraudDetector` with `@candidate` and `@champion` aliases) and a fully reproducible 5-stage **DVC Pipeline** (`dvc.yaml`).

Phase 9 implements the FastAPI inference service in `app/`. It loads the validated Champion pipeline directly from MLflow, applies the existing feature engineering, and uses the selected `0.90` operating threshold. The API does not load the local Joblib model artifact.

## Phase 10 — Docker Containerization

The inference-only Docker image packages the FastAPI service without the large raw dataset, training artifacts, notebooks, reports, tests, or local MLflow store. Compose mounts the existing local `mlruns/` tracking metadata read-only and exposes the same directory at the absolute artifact path recorded by the local registry. That second mount is writable because MLflow generates `registered_model_meta` beside the artifact while loading the registered model. This is intended for local development; it does not introduce a cloud MLflow server.

### Build

```bash
docker build -t credit-card-fraud-api:phase10 .
```

### Run

```bash
docker compose up -d
```

### Check containers and logs

```bash
docker compose ps
docker compose logs api
```

### API

`http://127.0.0.1:8000`

Swagger: `http://127.0.0.1:8000/docs`  
ReDoc: `http://127.0.0.1:8000/redoc`

### Stop

```bash
docker compose down
```

The container runs the existing inference flow only:

```text
Docker Container
    ↓
FastAPI Application
    ↓
Model Loader
    ↓
MLflow Registry
    ↓
CreditCardFraudDetector@champion
    ↓
Prediction → Fraud Probability → Threshold 0.90 → Fraud / Not Fraud
```

## Phase 9 — FastAPI Model Deployment

### Start API

```bash
uvicorn app.main:app --reload
```

Swagger UI is available at `/docs`; ReDoc is available at `/redoc`.

### Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Report service and Champion load status |
| POST | `/predict` | Predict one transaction |
| GET | `/model-info` | Return non-sensitive Champion metadata |

Champion model: `CreditCardFraudDetector@champion`  
Operating threshold: `0.90`

Example request:

```json
{
    "type": "TRANSFER",
    "amount": 1000.0,
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0
}
```

### Phase 9 Architecture

```text
Client
    ↓
FastAPI
    ↓
MLflow Model Registry
    ↓
CreditCardFraudDetector@champion
    ↓
Existing feature engineering + fitted preprocessing
    ↓
XGBoost
    ↓
Fraud probability
    ↓
Threshold 0.90
    ↓
Fraud / Not Fraud
```

---

## Phase 8 — MLflow Model Registry & Reproducible DVC Pipeline

### 1. MLflow Model Registry Lifecycle

Registered model name: `CreditCardFraudDetector`

Lifecycle flow:
```text
Experiment Run ──► Candidate Model ──► Validation (ModelValidator) ──► Champion Model (@champion)
```

- **Registry Module**: `src/models/registry.py` provides model registration, tag tracking, and alias management.
- **Model Validation**: `src/models/validate.py` enforces metric checks, strategy validation (`scale_weight_only`), operating threshold check (`0.90`), model loadability, and inference verification prior to Champion promotion.
- **Model URI**: Downstream components can load the validated champion model via:
  ```python
  import mlflow
  model = mlflow.sklearn.load_model("models:/CreditCardFraudDetector@champion")
  ```

### 2. Reproducible DVC Pipeline (`dvc.yaml`)

5-Stage Reproducible Pipeline:
```text
validate (scripts/validate_data.py)
   ↓
preprocess (scripts/preprocess_data.py)
   ↓
features (scripts/create_features.py)
   ↓
train (scripts/train_model.py)
   ↓
evaluate (scripts/evaluate_and_register.py)
```

Key DVC commands:
```bash
# Display pipeline DAG
dvc dag

# Check pipeline tracking status
dvc status

# Reproduce full pipeline end-to-end
dvc repro
```

---

## Dataset & Versioning

### Dataset Details
* **Raw Dataset**: `AIML DATASET.csv` (`data/raw/AIML DATASET.csv`, `6,362,620` rows × `11` columns)
* **Preprocessed Dataset**: `preprocessed.parquet` (`data/processed/preprocessed.parquet`, `142.49 MB`)
* **Cleaned Dataset**: `cleaned.parquet` (`data/processed/cleaned.parquet`, `6,362,620` rows × `10` columns, `249.57 MB`)
* **Trained Model Artifact**: `models/xgboost_fraud_model.joblib` (`0.57 MB`)

### Pipeline Execution Commands

```bash
# 1. Reproduce full pipeline end-to-end via DVC
dvc repro

# 2. View pipeline DAG graph
dvc dag

# 3. Check DVC pipeline status
dvc status

# 4. Launch MLflow UI to inspect Model Registry and experiments
mlflow ui
# Then open http://127.0.0.1:5000 in your browser

# 5. Run full pytest test suite (36 passed)
pytest -q
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
│   │   ├── evaluate.py     # Model evaluation module
│   │   ├── tracking.py     # MLflow experiment tracking helper
│   │   └── compare.py      # Model comparison, threshold analysis & selection
│   └── monitoring/         # Data & model drift detection modules
│
├── app/                    # FastAPI application & API endpoints
├── tests/                  # Pytest unit & integration tests
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   ├── test_model_training.py
│   ├── test_model_evaluation.py
│   ├── test_mlflow_tracking.py
│   └── test_model_comparison.py
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
│       ├── precision_recall_curve.png
│       ├── experiment_comparison.json
│       ├── experiment_comparison.png
│       ├── model_metadata.json
│       └── model_selection.md
│
├── scripts/                # Execution scripts
│   ├── validate_data.py
│   ├── run_eda.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── compare_models.py
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
python scripts/compare_models.py
pytest tests/
```
