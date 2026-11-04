"""Model Performance page: metrics, retraining, limitations."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import MODEL_PATH, load_model
from src.pipeline import train_and_save

from . import ui


def render() -> None:
    ui.hero("Model Performance", "How well does the Isolation Forest separate attacks from normal activity?")
    a = ui.get_analysis()
    bundle = st.session_state.bundle
    ev = a["evaluation"]

    st.subheader("Model")
    p = bundle.params
    ui.cards([
        ("Algorithm", "Isolation Forest", f"{p['n_estimators']} trees, contamination {p['contamination']}"),
        ("Trained on", f"{bundle.meta.get('n_train_rows', '?'):,} rows", f"at {bundle.meta.get('trained_at', '?')}"),
        ("Features", str(len(bundle.features)), ", ".join(bundle.features)),
    ])

    st.subheader("Evaluation on the active dataset")
    if not ev.get("available"):
        ui.note(ev.get("reason", "No ground truth available."), warn=True)
    else:
        ui.note("Metrics treat <b>Suspicious + High Risk</b> as a positive prediction. If the active dataset overlaps the "
                "training data the numbers are optimistic; the <b>held-out test metrics</b> below are the honest ones.")
        ui.cards([
            ("Precision", f"{ev['precision']:.1%}", "flagged that were real attacks"),
            ("Recall", f"{ev['recall']:.1%}", "attacks that were flagged"),
            ("F1 score", f"{ev['f1']:.2f}", "balance of the two"),
            ("ROC-AUC", f"{ev['roc_auc']:.3f}", "ranking quality of anomaly score"),
            ("False-positive rate", f"{ev['false_positive_rate']:.1%}", "normal records flagged"),
        ])
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            cm = ev["confusion"]
            fig = px.imshow(cm, text_auto=True, x=["Pred. normal", "Pred. flagged"], y=["Actual normal", "Actual attack"],
                            color_continuous_scale="Blues", title="Confusion matrix")
            fig.update_coloraxes(showscale=False)
            ui.show(fig, 320, "cm")
        with c2:
            fpr, tpr = ev["roc"]
            fig = go.Figure()
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name="Isolation Forest", line=dict(color=ui.ACCENT, width=3))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="#64748b"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            ui.show(fig, 320, "roc")
        if ev.get("per_type") is not None:
            fig = px.bar(ev["per_type"], x="attack_type", y="detection_rate", title="Detection rate per attack type",
                         color_discrete_sequence=[ui.ACCENT], text_auto=".0%")
            fig.update_layout(yaxis=dict(range=[0, 1.05], tickformat=".0%"))
            ui.show(fig, 300, "pertype")

    tm = bundle.meta.get("test_metrics")
    if tm:
        st.subheader("Held-out test metrics (recorded at training time)")
        ui.cards([
            ("Precision", f"{tm['precision']:.1%}", f"test rows: {tm['n']:,}"),
            ("Recall", f"{tm['recall']:.1%}", f"attacks in test: {tm['n_attacks']}"),
            ("F1", f"{tm['f1']:.2f}", ""),
            ("ROC-AUC", f"{tm['roc_auc']:.3f}", ""),
        ])

    st.subheader("Retrain the model")
    c1, c2, c3 = st.columns(3)
    cont = c1.slider("Expected anomaly share (contamination)", 0.01, 0.20, float(p["contamination"]), 0.01)
    trees = c2.slider("Number of trees", 50, 500, int(p["n_estimators"]), 50)
    c3.write("")
    if c3.button("Retrain and save", type="primary"):
        try:
            with st.spinner("Training..."):
                newb, m = train_and_save(st.session_state.df, cont, trees)
            ui.bump_model(newb)
            st.success(f"Model retrained and saved to {MODEL_PATH.name}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    if st.button("Reload saved model from disk"):
        try:
            ui.bump_model(load_model())
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Limitations - read before trusting the results")
    st.markdown(
        "- **Unusual is not malicious.** Backups, travel and night shifts look anomalous; stealthy attacks that blend in are missed.\n"
        "- **Garbage in, garbage out.** The model learns what is 'normal' from the training data; if attacks are already "
        "common in it, they look normal.\n"
        "- **Synthetic data.** Results on generated data are easier than real-world traffic and do not predict real performance.\n"
        "- **Static features.** Only eight per-record features are used; no payload content, user context or threat intelligence.\n"
        "- **No guarantees.** The system cannot predict every cyberattack or guarantee prevention; it supports analysts, it does not replace them.\n"
        "- **Concept drift.** Behaviour changes over time, so the model must be retrained periodically."
    )
