import streamlit as st
import json, os

SCREENSHOTS = 'screenshots'
INSIGHTS_PATH = os.path.join('models', 'feature_insights.json')


def render():
    st.title("🧠 Explainability & Feature Insights")
    st.markdown("Understanding *why* the model predicts failures — not just *what* it predicts.")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Analysis", "Business Interpretation"])

    def show_img(name, caption=""):
        path = os.path.join(SCREENSHOTS, name)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"Run `python -m src.explain` to generate {name}")

    with tab1:
        st.subheader("Random Forest Feature Importance")
        st.markdown("""
        Feature importance measures how much each input feature reduces impurity across all
        decision trees in the Random Forest.
        """)
        show_img("feature_importance.png", "Native Random Forest feature importances")

        if os.path.exists(INSIGHTS_PATH):
            with open(INSIGHTS_PATH) as f:
                insights = json.load(f)
            st.markdown("**Importance Scores:**")
            for item in insights:
                pct = item['importance'] * 100
                st.progress(min(pct / 30, 1.0), text=f"{item['feature']}: {pct:.1f}%")

    with tab2:
        st.subheader("SHAP Global Feature Impact")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for
        a specific prediction, based on game theory. Mean |SHAP| = global impact.
        """)
        show_img("shap_global.png", "Mean absolute SHAP values across test set")

        st.markdown("""
        **Interpreting SHAP:**
        - 🔴 High SHAP value → feature pushes prediction toward **failure**
        - 🔵 Low SHAP value → feature pushes prediction toward **no failure**
        - Magnitude = strength of influence
        """)

    with tab3:
        st.subheader("Business Interpretation")
        if os.path.exists(INSIGHTS_PATH):
            with open(INSIGHTS_PATH) as f:
                insights = json.load(f)

            for item in insights:
                with st.expander(f"📌 {item['feature']} — Importance: {item['importance']:.4f}"):
                    st.markdown(f"**Business Meaning:** {item['interpretation']}")
        else:
            st.info("Run `python -m src.explain` to generate insights.")

        st.markdown("---")
        st.markdown("""
        ### 💡 Maintenance Decision Logic
        | Feature | Threshold | Action |
        |---|---|---|
        | Tool Wear | > 200 min | Schedule immediate replacement |
        | Torque | > 60 Nm | Reduce load or inspect bearings |
        | Rotational Speed | < 1200 RPM | Check motor and drive |
        | Process Temp | > 313 K | Inspect cooling system |
        """)

