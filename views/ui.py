"""Shared UI helpers: styling, session state, cached analysis, small widgets."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import ModelBundle
from src.pipeline import analyze, ensure_sample_and_model
from src.risk_scoring import RiskConfig

NAVY = "#0a0a0a"      # black (sidebar / hero banner)
PANEL = "#ffffff"
BORDER = "#e5e7eb"
ACCENT = "#111827"    # near-black for neutral chart elements
TEXT = "#111111"
MUTED = "#6b7280"
RED = "#dc2626"
DARK_RED = "#991b1b"

CLASS_COLORS = {"Normal": "#16a34a", "Suspicious": "#d97706", "High Risk": RED}
LEVEL_COLORS = {"Low": "#16a34a", "Medium": "#d97706", "High": RED, "Critical": DARK_RED}
PRIORITY_COLORS = {"Critical": DARK_RED, "High": RED, "Medium": "#d97706", "Low": "#16a34a"}

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background:{NAVY}; border-right:1px solid #262626; }}
[data-testid="stSidebar"] :is(p, span, label, h1, h2, h3, div) {{ color:#f5f5f5; }}
h1,h2,h3 {{ letter-spacing: .2px; }}
.cs-hero {{ background:{NAVY}; border-left:6px solid {RED}; border-radius:12px; padding:18px 22px; margin-bottom:18px; }}
.cs-hero h1 {{ margin:0; font-size:1.6rem; color:#ffffff !important; }}
.cs-hero p {{ margin:.25rem 0 0; color:#d1d5db !important; }}
.cs-card {{ background:{PANEL}; border:1px solid {BORDER}; border-top:3px solid {NAVY}; border-radius:12px;
  padding:14px 16px; height:100%; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
.cs-card .lbl {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }}
.cs-card .val {{ font-size:1.9rem; font-weight:700; line-height:1.2; }}
.cs-card .sub {{ color:{MUTED}; font-size:.8rem; }}
.cs-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; color:#ffffff; }}
.cs-note {{ background:#f3f4f6; border-left:4px solid {NAVY}; border-radius:6px; padding:10px 14px; margin:8px 0 14px; color:{TEXT}; }}
.cs-warn {{ background:#fef2f2; border-left:4px solid {RED}; border-radius:6px; padding:10px 14px; margin:8px 0 14px; color:{DARK_RED}; }}
.cs-rec {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:14px 18px; margin-bottom:12px;
  box-shadow:0 1px 3px rgba(0,0,0,.06); }}
.cs-rec h4 {{ margin:0 0 6px; }}
.cs-rec .k {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; margin-top:8px; }}
/* dangerous recommendations (red / dark-red badge) -> red title text */
.cs-rec .cs-badge[style*="{RED}"] ~ b, .cs-rec .cs-badge[style*="{DARK_RED}"] ~ b {{ color:{RED}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="cs-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"cs-warn" if warn else "cs-note"}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="cs-badge" style="background:{color}">{text}</span>'


def metric_card(label: str, value: str, sub: str = "", color: str = TEXT) -> str:
    return (
        f'<div class="cs-card"><div class="lbl">{label}</div>'
        f'<div class="val" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>'
    )


def cards(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(metric_card(*it), unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#d1d5db")
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#d1d5db")
    return fig


def show(fig: go.Figure, height: int = 340, key: str | None = None) -> None:
    st.plotly_chart(style_fig(fig, height), width="stretch", key=key)


def gauge(score: float, level: str, title: str = "Overall risk") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"{title} - {level}"},
            number={"font": {"color": LEVEL_COLORS[level]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": LEVEL_COLORS[level]},
                "bgcolor": PANEL,
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 80], "color": "#fecaca"},
                    {"range": [80, 100], "color": "#fca5a5"},
                ],
            },
        )
    )
    return fig


# ----------------------------- session state -----------------------------

def init_state() -> None:
    """Load the sample dataset + model once per browser session."""
    if "df" in st.session_state:
        return
    try:
        df, bundle = ensure_sample_and_model()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not initialise the sample dataset/model: {exc}")
        st.stop()
    st.session_state.df = df
    st.session_state.bundle = bundle
    st.session_state.dataset_name = "Synthetic sample dataset"
    st.session_state.cfg = RiskConfig()
    st.session_state.ds_version = 0
    st.session_state.model_version = 0


def bump_dataset(df: pd.DataFrame, name: str) -> None:
    st.session_state.df = df
    st.session_state.dataset_name = name
    st.session_state.ds_version += 1


def bump_model(bundle: ModelBundle) -> None:
    st.session_state.bundle = bundle
    st.session_state.model_version += 1


def get_analysis() -> dict:
    """Return the full analysis; recomputed only when data/model/config change."""
    init_state()
    key = (
        st.session_state.ds_version,
        st.session_state.model_version,
        json.dumps(st.session_state.cfg.to_dict(), sort_keys=True, default=str),
    )
    if st.session_state.get("_analysis_key") != key:
        with st.spinner("Analysing activity..."):
            st.session_state._analysis = analyze(st.session_state.df, st.session_state.bundle, st.session_state.cfg)
        st.session_state._analysis_key = key
    return st.session_state._analysis
