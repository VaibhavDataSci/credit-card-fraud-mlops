# Exploratory Data Analysis (EDA) Findings & Insights

## 1. Dataset Overview

* **Source File**: `data/raw/AIML DATASET.csv` (PaySim Synthetic Financial Transactions)
* **Dataset Size**: `6,362,620` rows × `11` columns (`~470.67 MB`)
* **Data Completeness**: `0` missing (null) values across all columns.
* **Duplicate Records**: `0` duplicate rows detected.
* **Feature Breakdown**:
  * **Time**: `step` (Simulation hour index, 1 to 743).
  * **Categorical**: `type` (5 unique transaction types), `nameOrig` (Sender ID), `nameDest` (Recipient ID).
  * **Numerical Balances & Amounts**: `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`.
  * **Targets & Flags**: `isFraud` (Ground truth fraud label), `isFlaggedFraud` (System rule flag for transfers > 200k).

---

## 2. Class Imbalance Findings

* **Non-Fraud (`0`)**: `6,354,407` transactions (`99.871%`)
* **Fraud (`1`)**: `8,213` transactions (`0.129%`)
* **Imbalance Ratio**: `~773.7 : 1` (Extreme Class Imbalance)

> [!IMPORTANT]
> Because fraud accounts for less than **0.13%** of all transactions, naive accuracy models predicting all transactions as non-fraud would yield **99.87% accuracy** while failing entirely to catch fraud. Precision, Recall, F1-Score, and PR-AUC are mandatory metrics.

---

## 3. Transaction Type Patterns & Fraud Localization

Analysis of fraud distribution across the 5 transaction types yields a critical insight:

| Transaction Type | Total Transactions | Fraud Count (`isFraud=1`) | Fraud Rate (%) |
| :--- | :--- | :--- | :--- |
| `CASH_OUT` | 2,237,500 | 4,116 | 0.184% |
| `TRANSFER` | 532,909 | 4,097 | 0.769% |
| `PAYMENT` | 2,151,495 | 0 | 0.000% |
| `CASH_IN` | 1,399,284 | 0 | 0.000% |
| `DEBIT` | 41,432 | 0 | 0.000% |

> [!NOTE]
> **Key Finding**: Fraudulent transactions occur **ONLY** in `TRANSFER` and `CASH_OUT` types. `PAYMENT`, `CASH_IN`, and `DEBIT` transactions contain zero fraud instances.

---

## 4. Transaction Amount Observations

* **Non-Fraud Amounts**: Median transaction amount is `~$74,870`, with extreme outliers up to `$92.4 Million`.
* **Fraud Amounts**: Mean transaction amount is `~$1,467,908` (significantly higher than typical non-fraud transactions).
* **Pattern**: Fraudulent agents target high-value transfers to maximize illicit payout per transaction.

---

## 5. Balance Dynamics & Anomaly Patterns

1. **Complete Account Draining**:
   * In a large proportion of fraudulent `TRANSFER` and `CASH_OUT` actions, `oldbalanceOrg` equals `amount`, leaving `newbalanceOrig == 0.0`.
2. **Recipient Balance Anomalies**:
   * For many fraudulent `CASH_OUT` transactions, `oldbalanceDest` and `newbalanceDest` remain `0.0`, indicating fraudulent accounts or immediate external off-ramping.
3. **`isFlaggedFraud` Rule Inefficiency**:
   * Only **16 transactions** out of 6,362,620 are marked with `isFlaggedFraud == 1` (rule triggers when transfer > 200,000 in a single step). The baseline rule misses 8,197 fraud cases (99.8% false negative rate for the rule).

---

## 6. Considerations for Future Preprocessing Phase

* **Filtering Scope**: Restricting modeling dataset to `TRANSFER` and `CASH_OUT` transaction types will drop ~56% of non-fraud noise while retaining 100% of fraud instances.
* **High-Cardinality Identifiers**: `nameOrig` and `nameDest` contain millions of unique string IDs and should be dropped prior to modeling.
* **Engineered Features**:
  * `balance_diff_orig` = `oldbalanceOrg` - `newbalanceOrig`
  * `balance_diff_dest` = `newbalanceDest` - `oldbalanceDest`
  * `amount_to_balance_ratio` = `amount` / (`oldbalanceOrg` + 1)
* **Categorical Encoding**: Apply One-Hot Encoding to the `type` feature (`drop="first"`).

---

## 7. Considerations for Future Modeling Phase

* **Train/Test Splitting**: Stratified splitting (`stratify=y`) is required to preserve the 0.129% fraud distribution in train and test sets.
* **SMOTE Oversampling**: Apply SMOTE strictly to the training split inside an pipeline (`imbalanced-learn`) to prevent data leakage.
* **Hyperparameter Tuning**: Scale positive weight (`scale_pos_weight`) in XGBoost to handle residual class imbalance.
