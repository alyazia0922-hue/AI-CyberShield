"""Dataset Management page: generate synthetic data or upload a CSV."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_generator import generate_dataset
from src.preprocessing import COLUMN_DOC, MAX_ROWS, clean_dataframe

from . import ui

MAX_MB = 25


def _apply(raw: pd.DataFrame, name: str) -> None:
    df, rep = clean_dataframe(raw)
    if not rep.ok:
        for e in rep.errors:
            st.error(e)
        return
    for w in rep.warnings:
        st.warning(w)
    ui.bump_dataset(df, name)
    st.success(f"Loaded **{name}**: {rep.rows_out:,} valid rows (of {rep.rows_in:,}). Open any page to see the analysis.")


def render() -> None:
    ui.hero("Dataset Management", "Generate realistic synthetic data or upload your own CSV. No real sensitive data is needed.")
    ui.init_state()
    st.caption(f"Active dataset: **{st.session_state.dataset_name}** ({len(st.session_state.df):,} rows)")

    t1, t2, t3 = st.tabs(["Generate synthetic data", "Upload CSV", "Expected format"])

    with t1:
        c1, c2, c3, c4 = st.columns(4)
        n = c1.slider("Records", 500, 20000, 3000, 500)
        rate = c2.slider("Attack share %", 0, 30, 5) / 100
        users = c3.slider("Users", 5, 200, 40)
        seed = c4.number_input("Seed", 0, 99999, 42)
        if st.button("Generate and use this dataset", type="primary"):
            try:
                raw = generate_dataset(n_events=int(n), anomaly_rate=rate, n_users=int(users), seed=int(seed))
            except ValueError as exc:
                st.error(str(exc))
            else:
                _apply(raw, f"Synthetic ({n:,} records, {rate:.0%} attacks, seed {seed})")
        st.caption("Attacks are fabricated rows (brute force, port scan, suspicious login, exfiltration, traffic surge), "
                   "some deliberately subtle, plus benign look-alikes such as backups. Ground-truth labels are included "
                   "so model quality can be measured.")

    with t2:
        st.markdown(f"Upload a CSV (max {MAX_MB} MB, {MAX_ROWS:,} rows). Required columns are listed in *Expected format*. "
                    "Missing numeric values are filled with the median; negatives are clipped to 0.")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            if up.size > MAX_MB * 1024 * 1024:
                st.error(f"File is larger than {MAX_MB} MB.")
            else:
                try:
                    raw = pd.read_csv(up)
                except Exception as exc:
                    st.error(f"Could not read the file as CSV: {exc}")
                else:
                    st.write("Preview:")
                    st.dataframe(raw.head(10), width="stretch", hide_index=True)
                    if st.button("Validate and use this file", type="primary"):
                        _apply(raw, up.name)

    with t3:
        st.dataframe(COLUMN_DOC, width="stretch", hide_index=True)
        template = generate_dataset(n_events=100, seed=1).head(5)
        st.markdown("**Example rows**")
        st.dataframe(template, width="stretch", hide_index=True)
        st.download_button("Download template CSV", template.to_csv(index=False).encode(), "cybershield_template.csv", "text/csv")

    st.subheader("Current data preview")
    st.dataframe(st.session_state.df.head(200), width="stretch", hide_index=True)
