"""Business Insights page — operational metrics and ROI analysis."""
import os
import json
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def render():
    st.title("📈 Business Insights")
    st.caption("Operational and financial impact of the predictive maintenance model.")

    # Load quality report for base stats
    qr_path = os.path.join(PROCESSED_DIR, "quality_report.json")
    if not os.path.exists(qr_path):
        from src.preprocess import load_data, preprocess, validate_data
        df = load_data()
        _, _, _, _ = preprocess(df)
        with open(qr_path) as f:
            qr = json.load(f)
    else:
        with open(qr_path) as f:
            qr = json.load(f)

    failure_rate = qr.get("failure_rate", 0.034)
    total = qr.get("rows", 10000)
    failures = int(total * failure_rate)

    # Business parameters (user adjustable)
    st.markdown("### ⚙️ Business Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        cost_per_failure = st.number_input("Cost per unplanned failure ($)", 5000, 500000, 50000, 5000)
    with col2:
        cost_per_maintenance = st.number_input("Cost per preventive maintenance ($)", 500, 20000, 2000, 500)
    with col3:
        model_precision = st.slider("Model Precision (true alarm rate)", 0.5, 1.0, 0.82, 0.01)

    detection_rate = st.slider("Detection Rate (model recall)", 0.5, 1.0, 0.85, 0.01)

    st.markdown("---")
    st.markdown("### 💰 ROI Analysis")

    detected_failures = int(failures * detection_rate)
    false_alarms = int(detected_failures * (1 - model_precision) / model_precision)
    missed_failures = failures - detected_failures

    savings_from_detected = detected_failures * cost_per_failure
    cost_of_false_alarms = false_alarms * cost_per_maintenance
    cost_of_missed = missed_failures * cost_per_failure
    net_savings = savings_from_detected - cost_of_false_alarms - cost_of_missed

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failures Detected", f"{detected_failures:,}", delta=f"{detection_rate*100:.0f}% recall")
    c2.metric("False Alarms", f"{false_alarms:,}")
    c3.metric("Missed Failures", f"{missed_failures:,}", delta=f"-{missed_failures*cost_per_failure:,.0f}$", delta_color="inverse")
    c4.metric("💰 Net Savings", f"${net_savings:,.0f}", delta="vs reactive maintenance")

    # Cost waterfall chart
    st.markdown("### 📊 Cost-Benefit Breakdown")
    categories = ["Savings from\nDetected Failures", "Cost of\nFalse Alarms",
                  "Cost of\nMissed Failures", "Net Benefit"]
    values = [savings_from_detected, -cost_of_false_alarms, -cost_of_missed, net_savings]
    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Predictive Maintenance ROI Analysis", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value ($)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) * 0.01 if val >= 0 else min(values) * 0.01),
                f"${val:,.0f}", ha="center", fontsize=9,
                va="bottom" if val >= 0 else "top")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### ❓ Business Questions Answered")
    with st.expander("1. Which machines are most likely to fail?"):
        st.write("Use the Prediction page to score any machine in real-time by entering its sensor readings.")
    with st.expander("2. What operational factors are driving risk?"):
        st.write("Tool wear and torque are the top predictors. See the Explainability page for SHAP breakdowns.")
    with st.expander("3. How much downtime cost is exposed?"):
        st.write(f"Based on your dataset, {failures:,} failure events at ${cost_per_failure:,} each = ${failures*cost_per_failure:,} total exposure.")
    with st.expander("4. What maintenance action should be taken now?"):
        st.write("Each prediction returns a risk category (LOW / MEDIUM / HIGH / CRITICAL) and a recommended maintenance action.")

