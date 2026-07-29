"""Dashboard page — EDA charts and quality report."""
import os
import json
import streamlit as st
from PIL import Image

SCREENSHOTS = os.path.join(os.path.dirname(__file__), "..", "screenshots")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _load_img(name):
    path = os.path.join(SCREENSHOTS, name)
    if os.path.exists(path):
        return Image.open(path)
    return None


def _ensure_assets():
    if not os.path.exists(os.path.join(SCREENSHOTS, "histograms.png")):
        with st.spinner("Running EDA pipeline (first run)..."):
            from src.eda import run_eda
            run_eda()


def render():
    st.title("📊 Dashboard")
    st.caption("Exploratory data analysis and dataset quality metrics.")

    _ensure_assets()

    # Quality report
    qr_path = os.path.join(PROCESSED_DIR, "quality_report.json")
    if os.path.exists(qr_path):
        with open(qr_path) as f:
            qr = json.load(f)
        st.markdown("### 📋 Data Quality Report")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{qr['rows']:,}")
        c2.metric("Missing Values", qr["missing_values"])
        c3.metric("Duplicates", qr["duplicates"])
        c4.metric("Failure Rate", f"{qr['failure_rate']*100:.2f}%")
        st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Correlations", "📦 Failure Analysis"])

    with tab1:
        img = _load_img("histograms.png")
        if img:
            st.image(img, caption="Sensor Measurement Distributions", use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            img = _load_img("correlation_heatmap.png")
            if img:
                st.image(img, caption="Correlation Heatmap", use_container_width=True)
        with col2:
            img = _load_img("feature_correlation_analysis.png")
            if img:
                st.image(img, caption="Feature-to-Failure Correlation", use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            img = _load_img("failure_distribution.png")
            if img:
                st.image(img, caption="Failure Distribution", use_container_width=True)
        with col2:
            img = _load_img("boxplots.png")
            if img:
                st.image(img, caption="Feature Distributions by Failure Status", use_container_width=True)
