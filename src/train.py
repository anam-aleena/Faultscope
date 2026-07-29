"""Model training — Logistic Regression, Decision Tree, Random Forest."""
from __future__ import annotations

import json
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from src.preprocess import load_data, preprocess, split_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight="balanced",
        min_samples_leaf=2, random_state=42, n_jobs=-1),
}


def train() -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()
    X, y, le, schema = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    joblib.dump(le, os.path.join(MODELS_DIR, "preprocessor.joblib"))

    metrics = {}
    best_name, best_score, best_model = None, -1, None

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc": round(auc, 4),
        }
        if auc > best_score:
            best_score = auc
            best_name = name
            best_model = model

    metrics["best_model"] = best_name
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Best model: {best_name} | ROC-AUC: {best_score:.4f}")
    return metrics


if __name__ == "__main__":
    train()
