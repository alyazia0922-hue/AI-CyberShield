"""AI CyberShield - Predictive Cyber Risk Simulation Platform (entry point).

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="AI CyberShield", page_icon=":shield:", layout="wide",
                   initial_sidebar_state="expanded")

from views import (about, dashboard, dataset_management, model_performance, recommendations,  # noqa: E402
                   risk_assessment, simulation, threat_detection, ui)

PAGES = {
    "Overview": dashboard.render,
    "Threat Detection": threat_detection.render,
    "Risk Assessment": risk_assessment.render,
    "Attack Simulation": simulation.render,
    "AI Recommendations": recommendations.render,
    "Dataset Management": dataset_management.render,
    "Model Performance": model_performance.render,
    "About the Project": about.render,
}


def main() -> None:
    ui.inject_css()
    ui.init_state()
    with st.sidebar:
        st.markdown("## 🛡️ AI CyberShield")
        st.caption("Predictive Cyber Risk Simulation Platform")
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(f"Dataset: {st.session_state.dataset_name}")
        st.caption(f"Records: {len(st.session_state.df):,}")
        st.caption("Synthetic data only - no real network activity.")
    try:
        PAGES[page]()
    except Exception as exc:  # last-resort guard so the UI never shows a raw traceback
        st.error(f"Something went wrong while rendering this page: {exc}")
        st.exception(exc)


main()
