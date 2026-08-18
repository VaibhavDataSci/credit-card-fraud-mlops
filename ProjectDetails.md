# Credit Card Fraud Detection — MLOps Project Specification

## 1. Project Goal

Build an end-to-end **Credit Card Fraud Detection MLOps pipeline** using XGBoost.

The project must demonstrate the complete ML lifecycle:

**Data → Validation → Preprocessing → Feature Engineering → SMOTE → Training → MLflow → Model Registry → FastAPI → Docker → CI/CD → Monitoring → Drift Detection → Retraining**

The goal is not just to build a model, but to create a **reproducible, deployable and continuously monitored ML system**.

---

## 2. Technology Stack

* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost
* imbalanced-learn / SMOTE
* Git + GitHub
* DVC — data and pipeline versioning
* MLflow — experiment tracking + model registry
* FastAPI — model serving
* Docker — containerization
* GitHub Actions — CI/CD
* Evidently or equivalent — data drift monitoring
* Pytest — testing

Keep the architecture simple. Do not introduce Kubernetes, Kafka, Spark, Airflow or cloud infrastructure unless required later.

---

## 3. Responsibility of Each Tool

### Git

Version control for:

* Source code
* Configuration
* Pipeline definitions
* Documentation

### DVC

Version control for:

* Raw/processed datasets
* Data pipeline
* Dataset lineage and reproducibility

Important commands:

```bash
dvc init
dvc add
dvc repro
dvc status
dvc diff
```

### MLflow

Use MLflow for:

* Experiment tracking
* Parameters
* Metrics
* Artifacts
* Model logging
* Model versions
* Model Registry
* Model lineage

Example metrics:

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

### FastAPI

Serve the approved model through:

```text
GET  /health
POST /predict
GET  /model-info
```

### Docker

Containerize the FastAPI application and its dependencies.

### GitHub Actions

Automate:

```text
Push/PR
→ Install dependencies
→ Run tests
→ Build Docker image
```

### Monitoring

Monitor:

* API health
* Latency
* Errors
* Prediction distribution
* Data quality
* Model performance
* Data drift

---

# 4. Project Architecture

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
MLflow Tracking
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
New MLflow Model Version
    ↓
Champion Model
```

---

# 5. Data Pipeline

Implement these stages:

### Stage 1 — Data Collection

Store raw dataset in:

```text
data/raw/
```

Raw data must remain unchanged.

Track the dataset using DVC.

### Stage 2 — Validation

Check:

* Schema
* Required columns
* Missing values
* Duplicates
* Data types
* Invalid values
* Target values
* Class distribution

Pipeline must fail if validation fails.

### Stage 3 — EDA

Analyze:

* Dataset size
* Fraud percentage
* Feature distributions
* Correlations
* Outliers
* Class imbalance

EDA can be performed in notebooks, but production logic must be Python scripts.

### Stage 4 — Preprocessing

Create reusable preprocessing code.

Ensure the same preprocessing is used during training and inference.

### Stage 5 — Feature Engineering

Create meaningful transaction features.

Keep feature engineering reproducible and testable.

### Stage 6 — SMOTE

Because fraud is highly imbalanced:

```text
Train/Test Split
       ↓
SMOTE only on training data
       ↓
Model Training
```

Never apply SMOTE before splitting.

---

# 6. Model Development

Use **XGBoost** as the primary model.

Store hyperparameters in:

```text
params.yaml
```

Track experiments in MLflow.

Each training run should log:

### Parameters

* learning_rate
* max_depth
* n_estimators
* subsample
* random_state
* SMOTE settings

### Metrics

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

### Artifacts

* Confusion matrix
* Classification report
* Feature importance
* Evaluation results
* Model

Accuracy must NOT be the only metric.

---

# 7. MLflow Model Lifecycle

Use:

```text
Training
   ↓
MLflow Run
   ↓
Evaluation
   ↓
Model Registry
   ↓
Candidate
   ↓
Validation
   ↓
Champion
   ↓
Deployment
```

Registered model name:

```text
CreditCardFraudDetector
```

Track model metadata:

* MLflow run ID
* Model version
* Git commit
* DVC dataset version
* Metrics
* Training timestamp

Use a `champion` alias where appropriate.

The API should load the approved/champion model rather than a manually copied `model.pkl`.

---

# 8. DVC Pipeline

Create:

```text
dvc.yaml
```

Pipeline:

```text
validate
   ↓
preprocess
   ↓
features
   ↓
train
   ↓
evaluate
```

The pipeline must be reproducible using:

```bash
dvc repro
```

The project should make it possible to trace:

```text
Git Commit
+
DVC Dataset Version
+
MLflow Run
+
MLflow Model Version
```

---

# 9. API

Create a FastAPI service.

### `/health`

Returns service status.

### `/predict`

Input:

```json
{
  "feature_1": 0.5,
  "feature_2": -1.2
}
```

Output:

```json
{
  "prediction": 1,
  "fraud_probability": 0.91,
  "model_version": "3"
}
```

### `/model-info`

Return:

* Model name
* Model version
* Alias
* MLflow run ID where possible

---

# 10. Docker

Create:

```text
Dockerfile
docker-compose.yml
```

The container should run the FastAPI service.

Configuration/secrets must come from environment variables.

Never commit credentials.

---

# 11. CI/CD

Create GitHub Actions workflows.

### CI

```text
Push / Pull Request
       ↓
Install dependencies
       ↓
Run tests
       ↓
Build Docker image
```

### CD

Where deployment is configured:

```text
Validated code
      ↓
Build
      ↓
Deploy
```

Never automatically deploy an unvalidated model.

---

# 12. Monitoring

Implement monitoring for:

### System

* API latency
* Request count
* Error rate

### Data

* Missing values
* Schema changes
* Feature distributions

### Model

* Prediction distribution
* Precision
* Recall
* F1
* False positives
* False negatives

---

# 13. Data Drift

Implement:

```text
Reference Dataset
        +
Current Production Data
        ↓
Drift Detection
        ↓
Drift Report
```

Monitor numerical/categorical feature distributions.

Use Evidently or another suitable library.

Drift is a **warning signal**, not automatic proof that the model has failed.

Configurable drift thresholds should determine when investigation/retraining is considered.

---

# 14. Retraining

Retraining can be triggered by:

* Significant data drift
* Model performance degradation
* Sufficient new labeled data
* Scheduled retraining

Workflow:

```text
Drift / Performance Issue
        ↓
New Data
        ↓
DVC Version
        ↓
Validation
        ↓
Training
        ↓
MLflow
        ↓
Evaluation
        ↓
New Model Version
        ↓
Compare with Champion
        ↓
Promote or Reject
```

Never replace the champion model without evaluation.

---

# 15. Testing

Create:

```text
tests/
```

Test:

* Data validation
* Feature engineering
* Model prediction
* API endpoints
* Invalid API input
* Drift detection
* Model loading

Use Pytest.

---

# 16. Recommended Repository Structure

```text
credit-card-fraud-mlops/
│
├── .github/workflows/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── monitoring/
├── app/
├── tests/
├── configs/
├── reports/
├── scripts/
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── projectdetails.md
```

---

# 17. Implementation Phases

Implement incrementally.

### Phase 1

Repository + virtual environment + dependencies

### Phase 2

Dataset + DVC

### Phase 3

Data validation + EDA

### Phase 4

Preprocessing + feature engineering

### Phase 5

SMOTE + XGBoost

### Phase 6

MLflow experiment tracking

### Phase 7

MLflow Model Registry

### Phase 8

DVC reproducible pipeline

### Phase 9

FastAPI

### Phase 10

Docker

### Phase 11

Testing

### Phase 12

GitHub Actions CI/CD

### Phase 13

Monitoring

### Phase 14

Data drift detection

### Phase 15

Retraining + model promotion

### Phase 16

End-to-end validation + documentation

---

# 18. Critical Rules

1. Do not commit datasets directly to Git.
2. Use DVC for dataset versioning.
3. Use MLflow for experiment/model tracking.
4. Do not apply SMOTE before train/test splitting.
5. Do not train the final model only inside notebooks.
6. Keep training and inference preprocessing identical.
7. Do not hard-code model versions.
8. Do not commit secrets.
9. Do not deploy an unvalidated model.
10. Do not treat accuracy as the only fraud metric.
11. Do not assume drift automatically means model failure.
12. Keep the project reproducible.
13. Do not overengineer the solution.

---

# 19. Final Demonstration

The completed project should demonstrate:

```text
1. DVC dataset versioning
2. dvc repro
3. MLflow experiments
4. MLflow metrics/artifacts
5. Model Registry
6. Champion model
7. FastAPI prediction
8. Docker deployment
9. Automated tests
10. GitHub Actions
11. Monitoring
12. Drift detection
13. Retraining
14. New model version
15. Champion vs candidate comparison
```

The final project should demonstrate:

**Git = Code**

**DVC = Data + Pipeline**

**MLflow = Experiments + Models**

**FastAPI = Serving**

**Docker = Packaging**

**GitHub Actions = Automation**

**Monitoring = Observability**

**Drift Detection = Detecting Data Changes**

**Retraining = Continuous Model Improvement**

The core MLOps principle is:

> **A machine learning project is not finished when the model is trained. It is finished when the model can be reproduced, deployed, monitored, evaluated, and continuously improved.**
