"""Shared UI helpers: styling, session state, cached analysis, small widgets."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import ModelBundle
from src.pipeline import analyze, ensure_sample_and_model
from src.risk_scoring import RiskConfig

NAVY = "#0a1628"
PANEL = "#0f1f3a"
BORDER = "#1e3a5f"
ACCENT = "#38bdf8"
TEXT = "#e2e8f0"
MUTED = "#8fa3bf"

CLASS_COLORS = {"Normal": "#22c55e", "Suspicious": "#f59e0b", "High Risk": "#ef4444"}
LEVEL_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}
PRIORITY_COLORS = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#f59e0b", "Low": "#22c55e"}

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0c1b33 0%,#081120 100%); border-right:1px solid {BORDER}; }}
h1,h2,h3 {{ letter-spacing: .2px; }}
.cs-hero {{ background: linear-gradient(135deg,#0f2747 0%,#0a1628 70%); border:1px solid {BORDER};
  border-radius:14px; padding:18px 22px; margin-bottom:18px; }}
.cs-hero h1 {{ margin:0; font-size:1.6rem; }}
.cs-hero p {{ margin:.25rem 0 0; color:{MUTED}; }}
.cs-card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; height:100%; }}
.cs-card .lbl {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }}
.cs-card .val {{ font-size:1.9rem; font-weight:700; line-height:1.2; }}
.cs-card .sub {{ color:{MUTED}; font-size:.8rem; }}
.cs-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; color:#0a1628; }}
.cs-note {{ background:#0d2340; border-left:4px solid {ACCENT}; border-radius:6px; padding:10px 14px; margin:8px 0 14px; color:{TEXT}; }}
.cs-warn {{ background:#2a1f0a; border-left:4px solid #f59e0b; border-radius:6px; padding:10px 14px; margin:8px 0 14px; }}
.cs-rec {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:14px 18px; margin-bottom:12px; }}
.cs-rec h4 {{ margin:0 0 6px; }}
.cs-rec .k {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; margin-top:8px; }}
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
    fig.update_xaxes(gridcolor="#16304f", zerolinecolor="#16304f")
    fig.update_yaxes(gridcolor="#16304f", zerolinecolor="#16304f")
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
                    {"range": [0, 30], "color": "#123524"},
                    {"range": [30, 60], "color": "#3a2e0f"},
                    {"range": [60, 80], "color": "#3d2410"},
                    {"range": [80, 100], "color": "#3d1414"},
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
