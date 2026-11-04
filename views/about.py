"""About page: architecture, method, use cases, limitations."""
from __future__ import annotations

import streamlit as st

from . import ui


def render() -> None:
    ui.hero("AI CyberShield", "Predictive Cyber Risk Simulation Platform - an AI + cybersecurity prototype.")
    st.markdown(
        """
### What it is
A working prototype that uses **machine learning** to spot unusual security activity, converts findings into an
explainable **0-100 risk score**, lets you **simulate attack scenarios** on synthetic data, and produces
**evidence-linked recommendations**.

### How it works
1. **Data** - synthetic activity records (or your own CSV): logins, failed logins, traffic, ports, data volume, time of day.
2. **Preprocessing** - validation, missing-value handling, feature engineering (off-hours flag, log-scaled volumes).
3. **Detection** - `StandardScaler` + **Isolation Forest**. Records that are easy to isolate get high anomaly scores.
   `Suspicious` = model outlier; `High Risk` = in the most extreme 1% of the training distribution.
4. **Risk scoring** - five configurable factors (failed logins, unusual access, network traffic, data transfer,
   anomaly frequency) combined with adjustable weights.
5. **Simulation** - inject synthetic attack records, re-run detection and compare risk before/after.
6. **Recommendations** - rule engine tied to the counted evidence (never random).

### Cybersecurity use cases
Early warning for credential attacks, reconnaissance, data theft and traffic surges; SOC alert triage and
prioritisation; security-awareness demos; testing detection coverage before deployment.

### Responsible-use notes
- Anomaly detection identifies *unusual patterns*; it does **not prove** an attack occurred.
- The platform cannot predict every attack or guarantee prevention.
- Simulations are purely synthetic and never touch a real network.
- Isolating endpoints must only be done by authorized staff when the evidence justifies it.
        """
    )
    st.subheader("Technology")
    st.code("Python | Streamlit | Pandas / NumPy | scikit-learn (IsolationForest) | Plotly | Joblib", language="text")
