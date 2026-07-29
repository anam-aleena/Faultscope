"""Explainability page — generates charts in memory, no disk writes needed."""
import os
import sys
import json
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

FEATURE_INTERPRETATIONS = {
    "Tool wear [min]": "Higher tool wear dramatically increases failure risk. Tools nearing 200 min need urgent replacement.",
    "Torque [Nm]": "Excessive torque indicates mechanical stress. Values above 60 Nm correlate with PWF and OSF failures.",
    "Rotational speed [rpm]": "Speeds below 1300 rpm indicate overload. Combined with high torque, this is the most dangerous condition.",
    "Air temperature [K]": "Ambient temperature affects thermal stress. Elevated readings amplify process temperature risks.",
    "Process temperature [K]": "High process-to-air temperature differential (>11.5K) triggers heat dissipation failures.",
    "Type": "Heavy (H) machines show higher failure rates due to higher operational loads and longer duty cycles.",
}


def _load_model():
    import joblib
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def _feature_importance_chart(model):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0]) if hasattr(model, "coef_") else np.ones(len(FEATURES))

    sorted_idx = np.argsort(importances)
    sorted_features = [FEATURES[i] for i in sorted_idx]
    sorted_importance = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(sorted_features)))
    ax.barh(sorted_features, sorted_importance, color=colors, edgecolor="white")
    ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for i, v in enumerate(sorted_importance):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def render():
    st.title("💡 Explainability & Feature Insights")
    st.caption("Understanding *why* the model predicts failures — not just *what* it predicts.")

    tab1, tab2 = st.tabs(["📊 Feature Importance", "📖 Business Interpretation"])

    with tab1:
        st.subheader("Random Forest Feature Importance")
        st.markdown("""
        Feature importance measures how much each input feature reduces impurity
        across all decision trees in the Random Forest.
        """)

        model = _load_model()
        if model is None:
            with st.spinner("Training model for the first time..."):
                from src.results_summary import run_all
                run_all()
                model = _load_model()

        if model:
            fig = _feature_importance_chart(model)
            st.pyplot(fig)
            plt.close()

            st.info("""
            **How to read this chart:**
            Higher importance = stronger influence on the model's failure predictions.
            Tool wear and torque are typically the strongest predictors of machine failure.
            """)
        else:
            st.error("Model not available. Please check the deployment logs.")

    with tab2:
        st.subheader("Business Interpretation of Features")
        st.markdown("What each sensor reading means for maintenance decisions:")
        for feat, desc in FEATURE_INTERPRETATIONS.items():
            with st.expander(f"🔧 {feat}"):
                st.write(desc)
