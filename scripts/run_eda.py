#!/usr/bin/env python3
"""Script to execute Exploratory Data Analysis (EDA) and save visualizations."""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure non-interactive backend for server/CLI environments
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid", palette="muted")


def generate_eda_reports():
    """Load dataset, generate 7 EDA plots, and save to reports/eda/."""
    raw_path = BASE_DIR / "data" / "raw" / "AIML DATASET.csv"
    output_dir = BASE_DIR / "reports" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("        EXECUTING EXPLORATORY DATA ANALYSIS       ")
    print("==================================================")
    print(f"Loading raw dataset from {raw_path}...")

    df = pd.read_csv(raw_path)
    print(f"Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Sample for memory-efficient plotting of dense distributions (seed=42)
    sample_df = df.sample(n=min(100000, len(df)), random_state=42)
    print(f"Using reproducible sample of {len(sample_df)} rows for complex plotting.")

    # 1. Class Distribution Plot
    print("Generating Plot 1: class_distribution.png...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["isFraud"].value_counts()
    sns.barplot(x=["Non-Fraud (0)", "Fraud (1)"], y=counts.values, ax=ax[0], palette=["#2ecc71", "#e74c3c"])
    ax[0].set_title("Class Counts (Linear Scale)")
    ax[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + (max(counts.values) * 0.01), f"{v:,}\n({v/len(df)*100:.3f}%)", ha="center")

    sns.barplot(x=["Non-Fraud (0)", "Fraud (1)"], y=counts.values, ax=ax[1], palette=["#2ecc71", "#e74c3c"])
    ax[1].set_yscale("log")
    ax[1].set_title("Class Counts (Log Scale)")
    ax[1].set_ylabel("Count (Log Scale)")

    plt.suptitle("Target Class Distribution (isFraud)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close()

    # 2. Transaction Type Distribution Plot
    print("Generating Plot 2: transaction_type_distribution.png...")
    plt.figure(figsize=(9, 5))
    type_counts = df["type"].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, palette="Blues_d")
    plt.title("Transaction Count by Type", fontsize=14, fontweight="bold")
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    for i, v in enumerate(type_counts.values):
        plt.text(i, v + (max(type_counts.values) * 0.01), f"{v:,}", ha="center")
    plt.tight_layout()
    plt.savefig(output_dir / "transaction_type_distribution.png", dpi=300)
    plt.close()

    # 3. Fraud Distribution by Transaction Type
    print("Generating Plot 3: fraud_by_transaction_type.png...")
    plt.figure(figsize=(10, 5))
    fraud_types = df.groupby(["type", "isFraud"]).size().unstack(fill_value=0)
    fraud_types.plot(kind="bar", stacked=True, color=["#3498db", "#e74c3c"], figsize=(10, 5))
    plt.title("Fraud vs Non-Fraud Count by Transaction Type (Log Scale)", fontsize=14, fontweight="bold")
    plt.yscale("log")
    plt.xlabel("Transaction Type")
    plt.ylabel("Count (Log Scale)")
    plt.legend(["Non-Fraud (0)", "Fraud (1)"])
    plt.tight_layout()
    plt.savefig(output_dir / "fraud_by_transaction_type.png", dpi=300)
    plt.close()

    # 4. Transaction Amount Distribution
    print("Generating Plot 4: transaction_amount_distribution.png...")
    plt.figure(figsize=(10, 5))
    sns.histplot(sample_df["amount"] + 1, bins=50, log_scale=True, kde=True, color="#9b59b6")
    plt.title("Transaction Amount Distribution (Log Scale, Sampled N=100,000)", fontsize=14, fontweight="bold")
    plt.xlabel("Transaction Amount (Log Scale)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "transaction_amount_distribution.png", dpi=300)
    plt.close()

    # 5. Fraud Amount Comparison
    print("Generating Plot 5: fraud_amount_comparison.png...")
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="isFraud", y=np.log1p(sample_df["amount"]), data=sample_df, palette=["#2ecc71", "#e74c3c"])
    plt.title("Log(1 + Amount) Comparison: Non-Fraud vs Fraud", fontsize=14, fontweight="bold")
    plt.xlabel("isFraud (0: Non-Fraud, 1: Fraud)")
    plt.ylabel("Log(1 + Amount)")
    plt.tight_layout()
    plt.savefig(output_dir / "fraud_amount_comparison.png", dpi=300)
    plt.close()

    # 6. Balance Analysis
    print("Generating Plot 6: balance_analysis.png...")
    plt.figure(figsize=(10, 5))
    sample_df_copy = sample_df.copy()
    sample_df_copy["balance_diff_orig"] = sample_df_copy["oldbalanceOrg"] - sample_df_copy["newbalanceOrig"]
    sns.boxplot(x="isFraud", y=np.log1p(sample_df_copy["balance_diff_orig"].abs()), data=sample_df_copy, palette=["#34495e", "#e74c3c"])
    plt.title("Absolute Sender Balance Reduction Log(|oldbalanceOrg - newbalanceOrig| + 1)", fontsize=14, fontweight="bold")
    plt.xlabel("isFraud (0: Non-Fraud, 1: Fraud)")
    plt.ylabel("Log(|Balance Diff| + 1)")
    plt.tight_layout()
    plt.savefig(output_dir / "balance_analysis.png", dpi=300)
    plt.close()

    # 7. Correlation Matrix
    print("Generating Plot 7: correlation_matrix.png...")
    num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
    corr = df[num_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Numerical Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300)
    plt.close()

    print("==================================================")
    print(f"[SUCCESS] All 7 EDA plots successfully saved to {output_dir}")
    print("==================================================")


if __name__ == "__main__":
    generate_eda_reports()
