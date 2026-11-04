"""AI Recommendations page."""
from __future__ import annotations

import streamlit as st

from . import ui


def render() -> None:
    ui.hero("AI Recommendations", "Security actions generated from the findings in the current dataset.")
    a = ui.get_analysis()
    recs = a["recommendations"]
    ui.note("<b>How these are produced:</b> each recommendation comes from a rule that checks real detections "
            "(e.g. 'many records with >=5 failed logins' -> MFA + lockout). The evidence that triggered it is shown, "
            "so every suggestion can be audited. Containment steps require authorization - anomalies are not proof of compromise.")

    pr = st.multiselect("Priority", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
    shown = recs[recs["priority"].isin(pr)]
    if shown.empty:
        st.info("No recommendations for the selected priorities.")
        return
    for _, r in shown.iterrows():
        title_color = ui.RED if r["priority"] in ("Critical", "High") else ui.TEXT
        st.markdown(
            f'<div class="cs-rec">{ui.badge(r["priority"], ui.PRIORITY_COLORS[r["priority"]])} '
            f'&nbsp;<b style="font-size:1.05rem;color:{title_color}">{r["title"]}</b>'
            f'<div class="k">Linked finding</div>{r["finding"]}'
            f'<div class="k">Recommended action</div>{r["action"]}'
            f'<div class="k">Why it matters</div>{r["why"]}'
            f'<div class="k">Most affected</div>{r["affected"]}</div>',
            unsafe_allow_html=True,
        )
    st.download_button("Download recommendations (CSV)", shown.to_csv(index=False).encode(),
                       "cybershield_recommendations.csv", "text/csv")
