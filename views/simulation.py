"""Attack Simulation page (synthetic data only)."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import LABELS
from src.attack_simulator import SCENARIOS, run_simulation
from src.recommendations import generate_recommendations

from . import ui


def render() -> None:
    ui.hero("Attack Simulation", "Test hypothetical attack scenarios on synthetic data and see how detection and risk react.")
    ui.note("<b>Safe by design:</b> the simulator only adds fabricated rows to a synthetic dataset. It does not run any "
            "attack, scan, or connect to any network.")
    ui.init_state()

    keys = list(SCENARIOS)
    c1, c2 = st.columns([1, 2])
    with c1:
        scen = st.selectbox("Attack scenario", keys, format_func=lambda k: SCENARIOS[k]["name"])
    with c2:
        st.write("")
        st.info(SCENARIOS[scen]["description"])

    sl = SCENARIOS[scen]["sliders"]
    p1, p2, p3, p4 = st.columns(4)
    lo, hi, dv, lbl = sl["events"]
    n_events = p1.slider(lbl, lo, hi, dv, key=f"ev_{scen}")
    lo, hi, dv, lbl = sl["severity"]
    severity = p2.slider(lbl, lo, hi, dv, 0.1, key=f"sev_{scen}")
    off_hours = p3.toggle("Occurs outside business hours", True, key=f"off_{scen}")
    seed = p4.number_input("Random seed", 0, 9999, 7, key=f"seed_{scen}")

    if st.button("Run simulation", type="primary"):
        try:
            st.session_state.sim = run_simulation(scen, st.session_state.bundle, n_events, severity, off_hours,
                                                  seed=int(seed), cfg=st.session_state.cfg)
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return
    res = st.session_state.get("sim")
    if res is None:
        st.caption("Choose a scenario, adjust the parameters and press **Run simulation**.")
        return

    name = SCENARIOS[res.scenario]["name"]
    st.subheader(f"Results - {name}")
    b, s = res.baseline_risk, res.simulated_risk
    n_susp_b = int((res.baseline["classification"].astype(str) != "Normal").sum())
    n_susp_s = int((res.simulated["classification"].astype(str) != "Normal").sum())
    delta = s["score"] - b["score"]
    ui.cards([
        ("Risk score", f"{b['score']:.0f} -> {s['score']:.0f}", f"+{delta:.1f} points | {s['level']}",
         ui.LEVEL_COLORS[s["level"]]),
        ("Flagged records", f"{n_susp_b} -> {n_susp_s}", "Suspicious + High Risk"),
        ("Injected attack records detected", f"{res.n_detected} / {res.n_injected}", f"{res.detection_rate:.0%} detection rate"),
        ("High-risk events", str(int((res.simulated['classification'].astype(str) == 'High Risk').sum())),
         f"baseline {int((res.baseline['classification'].astype(str) == 'High Risk').sum())}", ui.CLASS_COLORS["High Risk"]),
    ])
    st.write("")

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure()
        fig.add_bar(x=["Baseline", "With attack"], y=[b["score"], s["score"]],
                    marker_color=[ui.LEVEL_COLORS[b["level"]], ui.LEVEL_COLORS[s["level"]]],
                    text=[f"{b['score']:.0f}", f"{s['score']:.0f}"], textposition="outside")
        fig.update_layout(title="Estimated risk score", yaxis=dict(range=[0, 105], title="score"))
        ui.show(fig, 320, "sim_score")
    with g2:
        sim = res.simulated
        fig = px.scatter(sim, x="timestamp", y="anomaly_score", color=sim["classification"].astype(str),
                         symbol=sim["is_attack"].map({0: "baseline", 1: "injected"}),
                         color_discrete_map=ui.CLASS_COLORS, opacity=0.75, title="Anomaly score over time")
        fig.update_layout(legend_title_text="")
        ui.show(fig, 320, "sim_scatter")

    st.markdown("**Factor breakdown (baseline vs. attack)**")
    comp = b["breakdown"][["factor", "sub_score"]].merge(s["breakdown"][["factor", "sub_score"]], on="factor",
                                                         suffixes=(" baseline", " with attack"))
    fig = px.bar(comp.melt("factor", var_name="run", value_name="sub-score"), x="factor", y="sub-score", color="run",
                 barmode="group", color_discrete_sequence=["#64748b", ui.ACCENT])
    ui.show(fig, 300, "sim_factors")

    st.markdown("**Recommendations triggered by this scenario**")
    recs = generate_recommendations(res.simulated, s)
    for _, r in recs.head(4).iterrows():
        st.markdown(f'<div class="cs-rec">{ui.badge(r["priority"], ui.PRIORITY_COLORS[r["priority"]])} '
                    f'&nbsp;<b>{r["title"]}</b><div class="k">Evidence</div>{r["finding"]}'
                    f'<div class="k">Suggested action</div>{r["action"]}</div>', unsafe_allow_html=True)

    missed = res.n_injected - res.n_detected
    if missed:
        ui.note(f"{missed} injected record(s) were <b>not</b> flagged - a reminder that anomaly detection has blind spots.",
                warn=True)
