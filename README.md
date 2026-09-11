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
* **Retraining and Approval**: Drift investigation, candidate evaluation, and explicit human approval before any Champion change.

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
Phase 8 implemented **MLflow Model Registry** (`CreditCardFraudDetector` with `@candidate` and `@champion` aliases) and a fully reproducible 5-stage **DVC Pipeline** (`dvc.yaml`). The normal initial lifecycle ends at the validated Champion model served by FastAPI; human approval is reserved for drift-triggered retraining and candidate replacement.

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
Raw Dataset
    ↓
DVC — Data Versioning
    ↓
Data Validation
    ↓
Preprocessing + Feature Engineering
    ↓
Model Training (XGBoost)
    ↓
MLflow — Experiment Tracking
    ↓
Model Evaluation & Comparison
    ↓
MLflow Model Registry
    ↓
Champion Model
    ↓
Production API (FastAPI)
    ↓
Streamlit — Bank Employee UI
```

Human approval is not part of this normal initial training-to-production path. It is used only when drift leads to retraining and a new candidate needs promotion.

## Phase 12 — GitHub Actions CI/CD

The `CI` workflow in `.github/workflows/ci.yml` runs on pushes to `main` and pull requests targeting `main`. It installs the existing `requirements.txt`, compiles the application, runs `pytest -q`, builds the existing Docker image, and smoke-tests the container's health, model-info, Swagger, and ReDoc endpoints with a bounded readiness loop.

The workflow does not run DVC, process the raw dataset, retrain models, or modify the MLflow Champion alias. The local filesystem MLflow registry is intentionally not committed, so a clean GitHub runner verifies the container and API contract without fabricating a Champion prediction. When a local `mlruns/` artifact is available, the smoke test also verifies the real `/predict` response.

## Retraining and Model Promotion

Phase 15 provides a controlled workflow for new labeled data:

```text
New Data / Drift Alert
    ↓
DVC Versioning (`dvc add data/raw/new_data.csv`)
    ↓
Validation → Preprocessing → Feature Engineering
    ↓
Training → MLflow Candidate
    ↓
Candidate Evaluation → Champion Comparison
    ↓
Promotion Report → Human Approval → Promote / Reject
```

Run candidate creation and comparison with:

```bash
python scripts/retrain_model.py --new-data data/raw/new_data.csv
```

Promotion requires the explicit `--promote` flag and all configured gates in `params.yaml`: no recall or PR-AUC regression, a measurable minimum F1 improvement, valid probabilities, and no false-negative increase. The default threshold remains `0.90` and the strategy remains `scale_weight_only`.

Drift does not automatically replace the model. Candidate models must pass validation and comparison before entering `PENDING_APPROVAL`. Human approval is required only in this drift/retraining path. Rejected candidates leave the current Champion unchanged, and previous Champion versions remain registered for rollback by an operator. Decisions are written to `reports/model/promotion_decision.json` and `reports/model/promotion_decision.md`.

## Human-in-the-Loop Model Promotion

Phase 16 adds an explicit operator gate:

```text
Drift → Investigation → Retraining → Candidate → Evaluation
    → Comparison → Promotion Report → Human Approval → Promote / Reject
```

Automated evaluation recommends whether a candidate is suitable, but the Champion is not changed until an operator explicitly approves the candidate. A passing candidate remains `PENDING_APPROVAL`; it is not promoted by retraining or API startup.

Review and approve a pending report with:

```bash
python scripts/approve_model.py --report reports/model/promotion_decision.json --operator <operator-name>
```

The CLI requires the exact input `APPROVE`; `yes`, `y`, `true`, `1`, and other values cancel safely. `REJECT` records the operator and reason, leaves the Champion unchanged, and retains the candidate for investigation. Approval records are appended to `reports/model/approval_history.json` and summarized in `reports/model/approval_decision.md`. Before promotion, candidate identity and the current Champion version are rechecked to prevent stale approvals. Previous Champion versions remain registered for rollback.

## Phase 17 — Prometheus + Grafana Monitoring

The observability stack is:

```text
FastAPI /metrics → Prometheus → Grafana
```

Prometheus metrics include request counters, request-duration histograms, 4xx/5xx errors, fraud/non-fraud prediction counters, fraud-probability buckets, data-quality errors, and low-cardinality Champion model information. The existing `/monitoring` endpoint remains available for the human-readable process-local summary.

Local URLs:

- FastAPI: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001` when host port `3000` is occupied

Grafana provisions a Prometheus datasource and the `Credit Card Fraud API` dashboard automatically. Dashboard panels cover request rate, error rate, p95 latency, fraud prediction rate, fraud/non-fraud predictions, probability distribution, data-quality errors, and Champion model information. Set `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` for local credentials; defaults are intended only for local development.

Monitoring is observational. Monitoring and Grafana alerts do not automatically retrain or promote models, and the Phase 16 human approval workflow remains the final promotion gate.

## Phase 14 — Drift Detection

Drift detection uses Evidently `0.7.21` through `src/monitoring/drift.py` and `scripts/detect_drift.py`. The reference is the existing DVC pipeline output `data/processed/cleaned.parquet`; drift compares the nine production feature columns and excludes the target `isFraud`. Reference and current data are bounded to a configurable sample (default `10,000` rows), so drift analysis does not process the full raw dataset.

Run a comparison with:

```bash
python scripts/detect_drift.py --current path/to/current.parquet
```

Reports are written to `reports/drift/drift_report.html` and `reports/drift/drift_summary.json`. Evidently's feature-level methods and `drift_share_threshold=0.5` determine whether overall dataset drift is detected. Missingness, schema, type, and finite-value checks run before comparison.

Drift detection is an investigation signal, not an automatic model replacement trigger. It does not retrain, register, promote, replace, or alter `CreditCardFraudDetector@champion`, its `scale_weight_only` strategy, or the `0.90` threshold. Reports are offline artifacts; no drift endpoint or prediction-path coupling was added.

## Phase 13 — Model/API Monitoring

The FastAPI service exposes `GET /monitoring` with lightweight aggregate metrics for:

- API latency
- Request count and response status
- Error count and error rate
- Fraud/non-fraud prediction distribution and probability statistics
- Predictions below or at/above the configured `0.90` threshold
- Data-quality validation errors by controlled category

Monitoring stores aggregate values in process-local memory only. It does not persist request bodies, account identifiers, `nameOrig`, or `nameDest`, and it does not use high-cardinality request values as labels. Counters reset when the process restarts, and multiple replicas would require a shared metrics backend. A future phase can connect this monitoring interface to Prometheus/Grafana or another centralized platform without changing prediction behavior.

---

## Phase 8 — MLflow Model Registry & Reproducible DVC Pipeline

### 1. MLflow Model Registry Lifecycle

Registered model name: `CreditCardFraudDetector`

Initial baseline registration:
```text
Validated Model
    ↓
MLflow Model Registry
    ↓
Champion Model
```

Drift / retraining:
```text
Drift Detection
    ↓
Investigation
    ↓
Retraining
    ↓
Candidate Model
    ↓
Evaluation
    ↓
PENDING_APPROVAL
    ↓
Human Approval
    ├── APPROVE → Promote Candidate to Champion
    └── REJECT → Keep Existing Champion
```

Human Approval is required only when drift leads to retraining and a candidate model needs promotion. It is not part of the initial training and registration flow.

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

## Overall Architecture

```text
Raw Dataset
    ↓
DVC — Data Versioning
    ↓
Data Validation
    ↓
Preprocessing + Feature Engineering
    ↓
Model Training (XGBoost)
    ↓
MLflow — Experiment Tracking
    ↓
Model Evaluation & Comparison
    ↓
MLflow Model Registry
    ↓
Champion Model
    ↓
Production API (FastAPI)
    ↓
Streamlit — Bank Employee UI
```

### Drift / Retraining / Approval Workflow

```text
Production Data / Predictions
    ↓
Drift Detection
    ↓
Drift Detected?
    ↓
Investigation
    ↓
Retraining
    ↓
Candidate Model
    ↓
Model Evaluation
    ↓
Human Approval
    ├── APPROVE → Promote Candidate to Champion
    └── REJECT → Keep Existing Champion
```

Human approval appears only in the drift/retraining workflow. It is not part of the normal initial training → registry → production path.

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
