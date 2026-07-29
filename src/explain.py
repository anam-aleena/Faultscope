"""Explainability — feature importance + SHAP (with graceful fallback)."""
from __future__ import annotations

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocess import load_data, preprocess, split_data, FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SCREENSHOTS = os.path.join(os.path.dirname(__file__), "..", "screenshots")

FEATURE_INTERPRETATIONS = {
    "Tool wear [min]": "Higher tool wear dramatically increases failure risk. Tools nearing 200 min need urgent replacement.",
    "Torque [Nm]": "Excessive torque indicates mechanical stress. Values above 60 Nm correlate with PWF and OSF failures.",
    "Rotational speed [rpm]": "Speeds below 1300 rpm indicate overload. Combined with high torque, this is the most dangerous condition.",
    "Air temperature [K]": "Ambient temperature affects thermal stress. Elevated readings amplify process temperature risks.",
    "Process temperature [K]": "High process-to-air temperature differential (>11.5K) triggers heat dissipation failures.",
    "Type": "Heavy (H) machines show higher failure rates due to higher operational loads and longer duty cycles.",
}


def run_explain():
    os.makedirs(SCREENSHOTS, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        from src.train import train
        train()

    model = joblib.load(model_path)
    df = load_data()
    X, y, le, _ = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 1. Native feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [FEATURES[i] for i in sorted_idx]
    sorted_importance = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(sorted_features)))
    ax.barh(sorted_features[::-1], sorted_importance[::-1], color=colors[::-1], edgecolor="white")
    ax.set_title("Feature Importance (Model Native)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for i, (feat, imp) in enumerate(zip(sorted_features[::-1], sorted_importance[::-1])):
        ax.text(imp + 0.001, i, f"{imp:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS, "feature_importance.png"), dpi=150)
    plt.close()

    # 2. SHAP global explanations
    try:
        import shap
        X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        fig, ax = plt.subplots(figsize=(9, 6))
        mean_shap = np.abs(shap_vals).mean(axis=0)
        sorted_idx2 = np.argsort(mean_shap)
        ax.barh([FEATURES[i] for i in sorted_idx2], mean_shap[sorted_idx2],
                color="#E74C3C", edgecolor="white", alpha=0.85)
        ax.set_title("SHAP Global Feature Importance", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOTS, "shap_global.png"), dpi=150)
        plt.close()

        # 3. SHAP local explanation (single instance)
        instance = X_test.iloc[:1]
        shap_local = explainer.shap_values(instance)
        local_vals = shap_local[1][0] if isinstance(shap_local, list) else shap_local[0]
        colors_local = ["#E74C3C" if v > 0 else "#3498DB" for v in local_vals]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(FEATURES, local_vals, color=colors_local, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("SHAP Local Explanation (Sample Instance)", fontsize=13, fontweight="bold")
        ax.set_xlabel("SHAP Value (impact on output)")
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOTS, "shap_local.png"), dpi=150)
        plt.close()
        print("SHAP explanations saved.")
    except Exception as e:
        print(f"SHAP unavailable ({e}). Saving fallback local explanation.")
        # Fallback: permutation-based local bar chart
        proba = model.predict_proba(X_test.iloc[:1])[0]
        fig, ax = plt.subplots(figsize=(9, 5))
        vals = sorted_importance * (1 if proba[1] > 0.5 else -1)
        colors_fb = ["#E74C3C" if v > 0 else "#3498DB" for v in vals]
        ax.barh(FEATURES[::-1], vals[::-1], color=colors_fb[::-1], edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Local Feature Contribution (Fallback)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Estimated Contribution")
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOTS, "shap_local.png"), dpi=150)
        plt.close()

    # Save interpretations
    import json
    interp_path = os.path.join(MODELS_DIR, "feature_interpretations.json")
    with open(interp_path, "w") as f:
        json.dump(FEATURE_INTERPRETATIONS, f, indent=2)

    print("Explainability assets saved.")


if __name__ == "__main__":
    run_explain()
