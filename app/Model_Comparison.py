"""Model Comparison page."""
import os
import json
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SCREENSHOTS = os.path.join(os.path.dirname(__file__), "..", "screenshots")


def _ensure_metrics():
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        with st.spinner("Training models (first run)..."):
            from src.train import train
            train()
    with open(metrics_path) as f:
        return json.load(f)


def _ensure_eval():
    eval_path = os.path.join(MODELS_DIR, "evaluation_report.json")
    if not os.path.exists(eval_path):
        with st.spinner("Running evaluation..."):
            from src.evaluate import evaluate
            evaluate()
    with open(eval_path) as f:
        return json.load(f)


def render():
    st.title("🔬 Model Comparison")
    st.caption("Side-by-side performance of Logistic Regression, Decision Tree, and Random Forest.")

    metrics = _ensure_metrics()
    best = metrics.get("best_model", "Random Forest")

    model_names = [k for k in metrics if k != "best_model"]
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    # Summary table
    st.markdown("### 📋 Performance Summary")
    st.markdown(f"**🏆 Best Model: {best}** (highest ROC-AUC)")

    cols = st.columns(len(model_names))
    for col, name in zip(cols, model_names):
        m = metrics[name]
        is_best = name == best
        header = f"{'⭐ ' if is_best else ''}{name}"
        with col:
            st.markdown(f"#### {header}")
            for mk, ml in zip(metric_keys, metric_labels):
                val = m[mk]
                col.metric(ml, f"{val:.4f}")
            st.markdown("---")

    # Comparison bar chart
    st.markdown("### 📊 Visual Comparison")
    x = np.arange(len(metric_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#3498DB", "#2ECC71", "#E74C3C"]
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [metrics[name][mk] for mk in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(SCREENSHOTS, exist_ok=True)
    fig.savefig(os.path.join(SCREENSHOTS, "model_comparison.png"), dpi=150)
    st.pyplot(fig)
    plt.close()

    # Confusion matrix + ROC
    eval_report = _ensure_eval()
    st.markdown("### 🔍 Best Model Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        cm_img = os.path.join(SCREENSHOTS, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(Image.open(cm_img), caption="Confusion Matrix", use_container_width=True)
    with col2:
        roc_img = os.path.join(SCREENSHOTS, "roc_curve.png")
        if os.path.exists(roc_img):
            st.image(Image.open(roc_img), caption="ROC Curve", use_container_width=True)
render = show
