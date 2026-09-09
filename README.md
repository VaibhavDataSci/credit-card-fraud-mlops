# Credit Card Fraud Detection — MLOps Pipeline

## Project Overview

This repository contains an end-to-end, production-ready **Credit Card Fraud Detection MLOps pipeline** built with XGBoost.

The objective of this project is to implement a robust, reproducible, deployable, and continuously monitored machine learning system for financial fraud detection.

The full MLOps workflow will incorporate:
* **Data Versioning**: DVC dataset tracking and remote storage reproducibility.
* **Data Validation**: Automated schema & missing data verification.
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
Current Development Phase: Phase 2 — Dataset + DVC Setup
```

Phase 1 established the repository foundation and environment configuration.
Phase 2 initializes **DVC** for data version control, configures a local development remote (`.dvc/storage`), sets `.gitignore` rules to exclude raw dataset binaries from Git while tracking `.dvc` metadata, and establishes dataset versioning workflows.

---

## Dataset & Versioning

### Dataset Details
* **Expected Filename**: `AIML DATASET.csv` (or `creditcard.csv`)
* **Location**: `data/raw/AIML DATASET.csv`
* **Format**: CSV
* **Domain & Purpose**: Synthetic financial transaction log (PaySim) for detecting fraudulent mobile money transactions (`isFraud`).
* **Columns**: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.

### Versioning Architecture
* **Git**: Tracks source code, configurations (`params.yaml`, `.env.example`), tests, documentation, and DVC metadata files (`*.dvc`, `.dvc/config`).
* **DVC**: Tracks raw data binaries in `data/raw/` and processed data outputs in `data/processed/`.
* **Git Exclusions**: Raw CSV dataset files are strictly ignored by `.gitignore` and managed exclusively by DVC.

### DVC Usage Commands

```bash
# Check dataset tracking status
dvc status

# Track a raw dataset file with DVC
dvc add data/raw/AIML\ DATASET.csv

# Push dataset version to DVC remote storage
dvc push

# Pull dataset version from DVC remote storage
dvc pull
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
│   ├── config
│   └── .gitignore
│
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
│
├── data/
│   ├── raw/                # Immutable raw datasets (tracked by DVC)
│   └── processed/          # Cleaned & transformed datasets (tracked by DVC)
│
├── notebooks/              # Exploratory notebooks (e.g. xgboost_experiments.ipynb)
│
├── src/                    # Core Python package
│   ├── __init__.py
│   ├── config.py           # Centralized environment variable configuration
│   ├── data/               # Data loading & validation modules
│   ├── features/           # Preprocessing & feature engineering modules
│   ├── models/             # Model training, SMOTE & evaluation modules
│   └── monitoring/         # Data & model drift detection modules
│
├── app/                    # FastAPI application & API endpoints
├── tests/                  # Pytest unit & integration tests
├── configs/                # Environment & deployment configurations
├── reports/                # Monitoring & evaluation reports
├── scripts/                # Utility & execution scripts
│
├── ProjectDetails.md       # Master project specification
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies (including dvc, python-dotenv)
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore patterns
└── params.yaml             # Pipeline and model hyperparameter configurations
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd credit-card-fraud-mlops
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
```

### 5. Fetch Dataset via DVC
```bash
dvc pull
```
