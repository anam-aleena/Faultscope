"""Model evaluation — confusion matrix, classification report, ROC."""
from __future__ import annotations

import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    ConfusionMatrixDisplay
)

from src.preprocess import load_data, preprocess, split_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SCREENSHOTS = os.path.join(os.path.dirname(__file__), "..", "screenshots")


def evaluate() -> dict:
    os.makedirs(SCREENSHOTS, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        from src.train import train
        train()

    model = joblib.load(model_path)
    df = load_data()
    X, y, le, _ = preprocess(df)
    _, X_test, _, y_test = split_data(X, y)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Failure", "Failure"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#E74C3C", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "roc_curve.png"), dpi=150)
    plt.close()

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    eval_report = {
        "roc_auc": round(roc_auc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(MODELS_DIR, "evaluation_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"Evaluation done. ROC-AUC: {roc_auc:.4f}")
    return eval_report


if __name__ == "__main__":
    evaluate()
