import streamlit as st
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import MachineInput, predict_failure

RISK_COLORS = {
    'LOW': '#22c55e',
    'MEDIUM': '#f59e0b',
    'HIGH': '#ef4444',
    'CRITICAL': '#7c3aed',
}

RISK_ICONS = {
    'LOW': '🟢',
    'MEDIUM': '🟡',
    'HIGH': '🔴',
    'CRITICAL': '🚨',
}


def show():
    st.title("🔮 Machine Failure Prediction")
    st.markdown("Enter machine sensor readings to get a real-time failure risk assessment.")

    if not os.path.exists(os.path.join('models', 'best_model.joblib')):
        st.error("Model not found. Run `python -m src.results_summary` first.")
        return

    with st.form("prediction_form"):
        st.subheader("⚙️ Machine Sensor Inputs")
        col1, col2 = st.columns(2)

        with col1:
            machine_type = st.selectbox(
                "Machine Type",
                options=['L', 'M', 'H'],
                help="L = Low quality, M = Medium, H = High quality"
            )
            air_temp = st.slider(
                "Air Temperature [K]", min_value=295.0, max_value=305.0,
                value=300.0, step=0.1
            )
            process_temp = st.slider(
                "Process Temperature [K]", min_value=305.0, max_value=315.0,
                value=310.0, step=0.1
            )

        with col2:
            rot_speed = st.slider(
                "Rotational Speed [RPM]", min_value=1000, max_value=2000,
                value=1500, step=10
            )
            torque = st.slider(
                "Torque [Nm]", min_value=3.0, max_value=80.0,
                value=40.0, step=0.5
            )
            tool_wear = st.slider(
                "Tool Wear [min]", min_value=0, max_value=253,
                value=100, step=1
            )

        submitted = st.form_submit_button("🚀 Predict Failure Risk", use_container_width=True)

    if submitted:
        try:
            machine = MachineInput(
                machine_type=machine_type,
                air_temperature=air_temp,
                process_temperature=process_temp,
                rotational_speed=rot_speed,
                torque=torque,
                tool_wear=tool_wear,
            )
            result = predict_failure(machine)

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            color = RISK_COLORS[result.risk_category]
            icon = RISK_ICONS[result.risk_category]

            col1, col2, col3 = st.columns(3)
            with col1:
                status = "⚠️ FAILURE" if result.failure_prediction else "✅ NORMAL"
                st.metric("Prediction", status)
            with col2:
                st.metric("Failure Probability", f"{result.failure_probability:.1%}")
            with col3:
                st.metric("Risk Level", f"{icon} {result.risk_category}")

            # Risk gauge bar
            prob_pct = int(result.failure_probability * 100)
            st.markdown(f"""
            <div style='background:#f1f5f9;border-radius:8px;padding:4px;margin:1rem 0;'>
                <div style='background:{color};width:{prob_pct}%;height:22px;border-radius:6px;
                     display:flex;align-items:center;justify-content:center;
                     color:white;font-weight:700;font-size:0.85rem;min-width:40px;
                     transition: width 0.5s ease;'>
                    {prob_pct}%
                </div>
            </div>""", unsafe_allow_html=True)

            # Recommended action
            st.markdown(f"""
            <div style='background:{color}18;border-left:4px solid {color};
                 border-radius:4px;padding:1rem;margin-top:0.5rem;'>
                <b>🔧 Recommended Action:</b><br>
                {result.recommended_action}
            </div>""", unsafe_allow_html=True)

            # Feature contributions
            if result.feature_contributions:
                st.markdown("---")
                st.subheader("📊 Feature Contributions")
                import pandas as pd
                contrib_df = pd.DataFrame(
                    list(result.feature_contributions.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                st.bar_chart(contrib_df.set_index('Feature'))

        except Exception as e:
            st.error(f"Prediction error: {e}")
            render = show
