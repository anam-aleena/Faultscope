"""Home page."""
import streamlit as st


def render():
    st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size:3rem; color:#E74C3C;'>🏭 PredictX</h1>
            <h3 style='color:#7F8C8D; font-weight:400;'>Industrial IoT Predictive Maintenance Platform</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🎯 Predict\nGet real-time failure probability for any machine configuration.")
    with col2:
        st.success("### 📊 Analyse\nExplore sensor telemetry, EDA charts, and model comparisons.")
    with col3:
        st.warning("### 💡 Explain\nUnderstand *why* the model flags a machine with SHAP insights.")

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    The platform follows a modular ML lifecycle:

    1. **IIoT Sensors** → machine type, air/process temperature, RPM, torque, tool wear
    2. **Data Pipeline** → ingestion, validation, quality reporting, feature encoding
    3. **Training Layer** → Logistic Regression, Decision Tree, **Random Forest** (best)
    4. **Evaluation Layer** → ROC-AUC, confusion matrix, classification report
    5. **Explainability** → Feature importance + SHAP global/local explanations
    6. **Application Layer** → This Streamlit dashboard
    """)

    st.markdown("---")
    st.markdown("### 📦 Dataset")
    st.markdown("""
    **AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repository)

    | Feature | Description |
    |---|---|
    | Type | Machine grade: L (light), M (medium), H (heavy) |
    | Air temperature [K] | Ambient temperature in Kelvin |
    | Process temperature [K] | Operational temperature in Kelvin |
    | Rotational speed [rpm] | Motor speed |
    | Torque [Nm] | Applied torque |
    | Tool wear [min] | Cumulative tool usage in minutes |
    | **Machine failure** | **Binary target: 0 = OK, 1 = Failure** |
    """)

    st.markdown("---")
    st.caption("PredictX · MIT License · Built with scikit-learn, SHAP, and Streamlit")
