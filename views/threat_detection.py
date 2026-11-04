"""Threat Detection page: anomaly detection results."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.anomaly_detection import LABELS

from . import ui


def render() -> None:
    ui.hero("Threat Detection", "Isolation Forest anomaly detection on user and network activity.")
    ui.note("<b>What this does:</b> Isolation Forest isolates records that are easy to separate from the crowd "
            "(few 'random cuts' needed). Those are <i>unusual</i>, not automatically malicious - e.g. a legitimate "
            "large backup also looks unusual. Analysts must confirm.")
    a = ui.get_analysis()
    scored = a["scored"]

    c1, c2, c3 = st.columns(3)
    classes = c1.multiselect("Classification", LABELS, default=["Suspicious", "High Risk"])
    min_score = c2.slider("Minimum anomaly score", 0, 100, 0)
    user = c3.selectbox("User", ["All"] + sorted(scored["user_id"].unique()))
    view = scored[scored["classification"].astype(str).isin(classes) & (scored["anomaly_score"] >= min_score)]
    if user != "All":
        view = view[view["user_id"] == user]

    ui.cards([
        ("Records shown", f"{len(view):,}", f"of {len(scored):,} analysed"),
        ("Highest anomaly score", f"{view['anomaly_score'].max():.0f}" if len(view) else "-", "0 = typical, 100 = extreme"),
        ("Distinct users", f"{view['user_id'].nunique():,}", "in this view"),
    ])
    st.write("")

    left, right = st.columns(2)
    with left:
        counts = scored["classification"].astype(str).value_counts().reindex(LABELS, fill_value=0).reset_index()
        counts.columns = ["Classification", "Events"]
        fig = px.pie(counts, names="Classification", values="Events", hole=0.55, color="Classification",
                     color_discrete_map=ui.CLASS_COLORS, title="Classification split")
        ui.show(fig, 340, "split")
    with right:
        samp = scored.sample(min(len(scored), 2500), random_state=1)
        fig = px.scatter(samp, x="connections", y="bytes_out_mb", color=samp["classification"].astype(str),
                         color_discrete_map=ui.CLASS_COLORS, log_x=True, log_y=True, opacity=0.7,
                         title="Connections vs outbound data (log scales)",
                         hover_data=["user_id", "failed_logins", "anomaly_score"])
        fig.update_layout(legend_title_text="")
        ui.show(fig, 340, "scatter")

    fig = px.histogram(scored, x="anomaly_score", color=scored["classification"].astype(str), nbins=50,
                       color_discrete_map=ui.CLASS_COLORS, title="Anomaly score distribution")
    fig.update_layout(legend_title_text="", barmode="stack")
    ui.show(fig, 280, "hist")

    st.subheader("Detected records")
    cols = ["timestamp", "user_id", "classification", "anomaly_score", "event_risk", "failed_logins",
            "connections", "unique_ports", "bytes_out_mb", "off_hours", "new_location", "reasons"]
    tbl = view.sort_values("anomaly_score", ascending=False)[cols].head(500).copy()
    tbl["classification"] = tbl["classification"].astype(str)
    st.dataframe(tbl, width="stretch", hide_index=True, column_config={
        "anomaly_score": st.column_config.ProgressColumn("anomaly_score", min_value=0, max_value=100, format="%.0f"),
        "event_risk": st.column_config.ProgressColumn("event_risk", min_value=0, max_value=100, format="%.0f"),
        "bytes_out_mb": st.column_config.NumberColumn("bytes_out_mb", format="%.1f"),
    })
    st.download_button("Download shown records (CSV)", view[cols].to_csv(index=False).encode(),
                       "cybershield_detections.csv", "text/csv")
    st.caption("Showing at most the 500 highest-scoring rows. 'reasons' lists which features are far above this "
               "dataset's normal range (robust z-score > 3).")
