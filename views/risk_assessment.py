"""Risk Assessment page: explainable, configurable 0-100 score."""
from __future__ import annotations

import copy

import plotly.express as px
import streamlit as st

from src.risk_scoring import FACTOR_HELP, FACTOR_LABELS, FACTORS, LEVEL_ORDER, RiskConfig

from . import ui


def render() -> None:
    ui.hero("Risk Assessment", "How the 0-100 cyber risk score is calculated - and how to tune it.")
    cfg: RiskConfig = st.session_state.cfg if "cfg" in st.session_state else RiskConfig()
    a = ui.get_analysis()
    cfg = st.session_state.cfg
    overall, scored = a["overall"], a["scored"]

    ui.cards([
        ("Overall risk score", f"{overall['score']:.1f}", f"{overall['level']} risk", ui.LEVEL_COLORS[overall["level"]]),
        ("Critical events", f"{int((scored['risk_level'].astype(str) == 'Critical').sum()):,}", "per-event level",
         ui.LEVEL_COLORS["Critical"]),
        ("High events", f"{int((scored['risk_level'].astype(str) == 'High').sum()):,}", "per-event level",
         ui.LEVEL_COLORS["High"]),
    ])
    st.write("")

    c1, c2 = st.columns([1, 1])
    with c1:
        ui.show(ui.gauge(overall["score"], overall["level"]), 300, "rg")
    with c2:
        bd = overall["breakdown"]
        fig = px.bar(bd, x="contribution", y="factor", orientation="h", text="contribution",
                     title="Contribution of each factor to the overall score", color_discrete_sequence=[ui.ACCENT])
        fig.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis_title="points (of 100)")
        ui.show(fig, 300, "rb")

    st.subheader("How the score is calculated")
    st.markdown(
        "1. For each factor we measure **what share of records** shows that behaviour.\n"
        "2. That share is divided by the factor's *saturation level* (share at which it counts as 100) and capped at 100 "
        "-> the **sub-score**.\n"
        "3. The overall score is the **weighted average** of the five sub-scores "
        "(`score = sum(weight x sub-score)`, weights normalised to 100%).\n"
        "4. **Levels:** 0-30 Low, 31-60 Medium, 61-80 High, 81-100 Critical."
    )
    bd = overall["breakdown"].drop(columns="key").rename(columns={"share_of_records_%": "share of records %",
                                                                    "weight_%": "weight %"})
    st.dataframe(bd, width="stretch", hide_index=True)

    st.subheader("Configure the scoring logic")
    ui.note("Changes apply instantly to every page. Weights are re-normalised so they always add up to 100%.")
    with st.form("risk_cfg"):
        new = copy.deepcopy(cfg)
        wc = st.columns(len(FACTORS))
        for col, f in zip(wc, FACTORS):
            new.weights[f] = col.slider(FACTOR_LABELS[f], 0, 100, int(round(cfg.weights[f] * 100)),
                                        help=FACTOR_HELP[f]) / 100
        st.markdown("**Saturation (% of records that makes a factor 'maxed out')**")
        sc = st.columns(len(FACTORS))
        for col, f in zip(sc, FACTORS):
            new.saturation[f] = col.number_input(FACTOR_LABELS[f], 1.0, 100.0, float(cfg.saturation[f] * 100),
                                                 step=1.0, key=f"sat_{f}") / 100
        t1, t2 = st.columns(2)
        new.failed_login_threshold = int(t1.number_input("Failed-login threshold", 2, 50, cfg.failed_login_threshold))
        new.z_threshold = float(t2.number_input("Abnormality threshold (robust z-score)", 2.0, 20.0, cfg.z_threshold, step=0.5))
        b1, b2 = st.columns(2)
        apply = b1.form_submit_button("Apply configuration", type="primary")
        reset = b2.form_submit_button("Reset to defaults")
    if apply:
        st.session_state.cfg = new
        st.rerun()
    if reset:
        st.session_state.cfg = RiskConfig()
        st.rerun()

    st.subheader("Per-event risk levels")
    dist = scored["risk_level"].astype(str).value_counts().reindex(LEVEL_ORDER, fill_value=0).reset_index()
    dist.columns = ["Level", "Events"]
    fig = px.pie(dist, names="Level", values="Events", hole=0.5, color="Level", color_discrete_map=ui.LEVEL_COLORS)
    ui.show(fig, 300, "rpie")
