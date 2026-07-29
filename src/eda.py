"""Exploratory data analysis — saves images to screenshots/."""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.preprocess import load_data, validate_data, preprocess, FEATURES, TARGET

SCREENSHOTS = os.path.join(os.path.dirname(__file__), "..", "screenshots")


def run_eda():
    os.makedirs(SCREENSHOTS, exist_ok=True)
    df = load_data()
    report = validate_data(df)
    print("Quality report:", report)

    numeric_features = [f for f in FEATURES if f != "Type"]

    # 1. Histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_features):
        axes[i].hist(df[col], bins=40, color="#4A90D9", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Sensor Measurement Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "histograms.png"), dpi=150)
    plt.close()

    # 2. Correlation heatmap
    corr_df = df[numeric_features + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # 3. Failure distribution
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["No Failure", "Failure"], counts.values, color=["#2ECC71", "#E74C3C"],
           edgecolor="white", width=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=11)
    ax.set_title("Machine Failure Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "failure_distribution.png"), dpi=150)
    plt.close()

    # 4. Boxplots by failure status
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_features):
        df.boxplot(column=col, by=TARGET, ax=axes[i], notch=False,
                   boxprops=dict(color="#4A90D9"),
                   medianprops=dict(color="#E74C3C", linewidth=2))
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("Machine Failure")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distribution by Failure Status", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "boxplots.png"), dpi=150)
    plt.close()

    # 5. Feature-to-failure correlation
    corrs = df[numeric_features].corrwith(df[TARGET]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in corrs.values]
    ax.barh(corrs.index, corrs.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Correlation with Machine Failure", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "feature_correlation_analysis.png"), dpi=150)
    plt.close()

    # 6. Statistical summary CSV
    df[numeric_features].describe().to_csv(
        os.path.join(SCREENSHOTS, "statistical_summary.csv"))

    print("EDA complete. Saved to screenshots/.")


if __name__ == "__main__":
    run_eda()
