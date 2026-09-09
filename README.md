# Credit Card Fraud Detection — MLOps Pipeline

## Project Overview

This repository contains an end-to-end, production-ready **Credit Card Fraud Detection MLOps pipeline** built with XGBoost.

The objective of this project is to implement a robust, reproducible, deployable, and continuously monitored machine learning system for financial fraud detection.

The full MLOps workflow will incorporate:
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
Current Development Phase: Phase 1 — Project Foundation
```

Phase 1 establishes the initial repository layout, directory hierarchy, baseline configurations, dependency list, `.gitignore` guardrails, and preserving exploratory Jupyter Notebooks (`notebooks/xgboost_experiments.ipynb`).

---

## Planned Architecture

```text
Data
 ↓
Validation
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
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
│
├── data/
│   ├── raw/                # Immutable raw datasets (tracked by DVC)
│   └── processed/          # Cleaned & transformed datasets
│
├── notebooks/              # Exploratory notebooks (e.g. xgboost_experiments.ipynb)
│
├── src/                    # Core Python package
│   ├── __init__.py
│   ├── data/               # Data loading & validation modules
│   ├── features/           # Preprocessing & feature engineering modules
│   ├── models/             # Model training, SMOTE & evaluation modules
│   └── monitoring/         # Data & model drift detection modules
│
├── app/                    # FastAPI application & API endpoints
│
├── tests/                  # Pytest unit & integration tests
│
├── configs/                # Environment & deployment configurations
├── reports/                # Monitoring & evaluation reports
├── scripts/                # Utility & execution scripts
│
├── ProjectDetails.md       # Master project specification
├── README.md               # Project documentation
├── requirements.txt        # Initial Python dependencies
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
