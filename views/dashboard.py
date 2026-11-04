"""Overview page: KPIs, risk distribution, timeline, top actions."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.anomaly_detection import LABELS
from src.risk_scoring import LEVEL_ORDER

from . import ui


def render() -> None:
    ui.hero("Security Overview", "Real-time-style summary of analysed activity, risk and recommended actions.")
    a = ui.get_analysis()
    scored, overall, recs = a["scored"], a["overall"], a["recommendations"]

    st.caption(f"Dataset: **{st.session_state.dataset_name}**  |  {len(scored):,} records  |  "
               f"{scored['timestamp'].min():%d %b %Y} - {scored['timestamp'].max():%d %b %Y}")

    # ---- filters
    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        d0, d1 = scored["timestamp"].min().date(), scored["timestamp"].max().date()
        rng = c1.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
        classes = c2.multiselect("Classification", LABELS, default=LABELS)
        users = c3.multiselect("Users", sorted(scored["user_id"].unique()), placeholder="All users")
    view = scored
    if isinstance(rng, tuple) and len(rng) == 2:
        view = view[(view["timestamp"].dt.date >= rng[0]) & (view["timestamp"].dt.date <= rng[1])]
    if classes:
        view = view[view["classification"].astype(str).isin(classes)]
    if users:
        view = view[view["user_id"].isin(users)]
    if view.empty:
        ui.note("No records match the current filters.", warn=True)
        return

    n_susp = int((view["classification"].astype(str) == "Suspicious").sum())
    n_high = int((view["classification"].astype(str) == "High Risk").sum())
    ui.cards([
        ("Activities analysed", f"{len(view):,}", "records in current view"),
        ("Suspicious", f"{n_susp:,}", f"{n_susp / len(view):.1%} of view", ui.CLASS_COLORS["Suspicious"]),
        ("High-risk events", f"{n_high:,}", f"{n_high / len(view):.1%} of view", ui.CLASS_COLORS["High Risk"]),
        ("Overall risk score", f"{overall['score']:.0f} / 100", overall["level"] + " risk",
         ui.LEVEL_COLORS[overall["level"]]),
    ])
    ui.note("<b>Reading this page:</b> the ML model flags <i>unusual</i> behaviour. A flag is a lead for an analyst to "
            "check - it does not prove an attack happened.")

    c1, c2 = st.columns([1, 1])
    with c1:
        ui.show(ui.gauge(overall["score"], overall["level"]), 300, "gauge")
    with c2:
        dist = view["risk_level"].astype(str).value_counts().reindex(LEVEL_ORDER, fill_value=0).reset_index()
        dist.columns = ["Risk level", "Events"]
        fig = px.bar(dist, x="Risk level", y="Events", color="Risk level", color_discrete_map=ui.LEVEL_COLORS,
                     title="Risk distribution (per-event risk level)", text="Events")
        fig.update_layout(showlegend=False)
        ui.show(fig, 300, "dist")

    t = view.assign(bucket=view["timestamp"].dt.floor("6h"))
    tl = t.groupby(["bucket", t["classification"].astype(str)], observed=True).size().reset_index(name="events")
    tl.columns = ["time", "classification", "events"]
    fig = px.bar(tl, x="time", y="events", color="classification", color_discrete_map=ui.CLASS_COLORS,
                 category_orders={"classification": LABELS}, title="Security activity timeline (6-hour buckets)")
    fig.update_layout(barmode="stack")
    ui.show(fig, 340, "timeline")

    st.subheader("Recommended security actions")
    for _, r in recs.head(4).iterrows():
        st.markdown(
            f'<div class="cs-rec">{ui.badge(r["priority"], ui.PRIORITY_COLORS[r["priority"]])} '
            f'&nbsp;<b>{r["title"]}</b><div class="k">Evidence</div>{r["finding"]}</div>',
            unsafe_allow_html=True,
        )
    st.caption("Full list with reasons: open **AI Recommendations** in the sidebar.")
