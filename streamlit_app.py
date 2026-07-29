"""Streamlit entry point for PredictX."""
from __future__ import annotations

import streamlit as st
from app import Business_Insights, Dashboard, Explainability, Home, Model_Comparison, Prediction

PAGES = {
    "🏠 Home": Home.render,
    "📊 Dashboard": Dashboard.render,
    "🎯 Prediction": Prediction.render,
    "🔬 Model Comparison": Model_Comparison.render,
    "💡 Explainability": Explainability.render,
    "📈 Business Insights": Business_Insights.render,
}


def main() -> None:
    st.set_page_config(page_title="PredictX", page_icon="🏭", layout="wide")
    st.sidebar.markdown("# 🏭 PredictX")
    st.sidebar.markdown("*Industrial IoT Predictive Maintenance*")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with scikit-learn · SHAP · Streamlit")
    PAGES[page]()


if __name__ == "__main__":
    main()
