---
title: AI CyberShield
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# AI CyberShield — Predictive Cyber Risk Simulation Platform

An interactive AI + cybersecurity prototype: it detects unusual security activity with an
**Isolation Forest**, turns findings into an explainable **0–100 risk score**, lets you
**simulate attack scenarios on synthetic data**, and generates **evidence-linked recommendations**.

> Anomaly detection finds *unusual* behaviour. It does not prove an attack happened, cannot predict
> every attack, and cannot guarantee prevention. All data is synthetic; the simulator never touches a network.

## Quick start

```bash
cd AI-CyberShield
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline            # (optional) regenerate sample data + retrain model
streamlit run app.py              # opens http://localhost:8501
```

The first launch creates `data/sample_dataset.csv` and `models/isolation_forest.joblib` automatically if missing.
Run the tests with `python -m pytest -q` (27 tests, includes rendering every page headlessly).

## Project structure

```
AI-CyberShield/
├── app.py                    # entry point: sidebar navigation + page routing
├── requirements.txt
├── .streamlit/config.toml    # dark-navy theme
├── data/sample_dataset.csv   # synthetic sample (3,000 records, labelled)
├── models/isolation_forest.joblib   # trained model bundle (joblib)
├── src/                      # all logic, no Streamlit imports
│   ├── data_generator.py     # synthetic normal + attack records (incl. stealth & benign look-alikes)
│   ├── preprocessing.py      # validation, cleaning, feature engineering, CSV schema
│   ├── anomaly_detection.py  # train / save / load / predict / explain (Isolation Forest)
│   ├── risk_scoring.py       # configurable 0-100 score + per-event risk
│   ├── attack_simulator.py   # 5 scenarios, injects synthetic rows, re-scores
│   ├── recommendations.py    # rule engine tied to counted evidence
│   ├── evaluation.py         # precision/recall/F1/ROC-AUC/confusion matrix
│   └── pipeline.py           # glue: analyze(), train_and_save(), CLI
├── views/                    # one Streamlit page per file (render() functions)
│   ├── dashboard.py  threat_detection.py  risk_assessment.py  simulation.py
│   ├── recommendations.py  dataset_management.py  model_performance.py  about.py
│   └── ui.py                 # CSS, cards, charts theme, session state, cached analysis
└── tests/                    # pytest: core logic + Streamlit AppTest for every page
```

`views/` is used instead of Streamlit's magic `pages/` folder so that navigation is controlled
by `app.py` (named sidebar, shared state) rather than auto-generated from file names.

## Input CSV format

| column | status | meaning |
|---|---|---|
| timestamp | required | date-time of the activity |
| user_id | required | pseudonymous account id |
| failed_logins | required | failed login attempts (>= 0) |
| bytes_out_mb / bytes_in_mb | required | data sent / received in MB |
| connections | required | connections opened |
| unique_ports | required | distinct destination ports |
| session_minutes | required | session length |
| source_ip, new_location (0/1) | optional | extra context (defaults applied) |
| is_attack (0/1), attack_type | optional | ground truth → enables evaluation |

Missing numbers are filled with the median, negatives clipped to 0, bad timestamps dropped, and every
adjustment is reported to the user.

## How the AI works

1. **Features (8):** failed logins, off-hours flag (22:00–05:59), new-location flag, log-scaled bytes out/in,
   connections, unique ports, session minutes.
2. **Model:** `StandardScaler → IsolationForest(n_estimators=200, contamination=0.05)`, trained **unsupervised**
   (labels are never used for training).
3. **Classification:** `Normal` (model inlier) · `Suspicious` (model outlier) · `High Risk` (decision score in the
   lowest 1% of the training distribution).
4. **Anomaly score 0–100:** the Isolation-Forest score rescaled between the training median and 99.9th percentile.
5. **Explanations:** per-record robust z-scores (median/MAD) list *why* it is unusual ("high outbound data volume, off-hours access").
6. **Evaluation:** a 70/30 stratified split; metrics are computed on the held-out 30% (see *Model Performance* page).

## Risk score

`score = Σ weight × sub-score` over five factors — failed logins, unusual access, network traffic,
data transfer, suspicious-activity frequency. A sub-score is the share of records showing the behaviour divided by a
configurable saturation level (capped at 100). Weights, saturation levels and thresholds are editable in the
**Risk Assessment** page. Levels: 0–30 Low · 31–60 Medium · 61–80 High · 81–100 Critical.

## Pages

Overview · Threat Detection · Risk Assessment · Attack Simulation · AI Recommendations · Dataset Management ·
Model Performance · About the Project.

## Demo guide (≈5 minutes)

1. **Overview** – KPIs, gauge, timeline; open *Filters* and pick one user.
2. **Threat Detection** – filter to *High Risk*; read the `reasons` column; note benign look-alikes among flags.
3. **Risk Assessment** – show the breakdown, then halve the failed-login weight and watch the score move.
4. **Attack Simulation** – run *Data Exfiltration* with intensity 2.0: score jumps, new recommendation appears; then run it with
   low intensity and point out any missed records.
5. **AI Recommendations** – show evidence → action → why for each item.
6. **Model Performance** – confusion matrix, ROC, per-attack-type detection, limitations.

## Limitations

- Unusual ≠ malicious (backups, travel, night shifts raise false positives); stealthy attacks are missed.
- Learns "normal" from its training data; pre-existing attacks in it look normal.
- Synthetic data is easier than real traffic, so reported metrics do not predict production performance.
- Only 8 per-record features: no payload content, user context, or threat intelligence.
- Needs periodic retraining (behaviour drifts) and human review of every alert.

## Cybersecurity use cases

SOC alert triage and prioritisation · early warning for credential attacks, reconnaissance and data theft ·
insider-threat screening · detection-coverage testing before deployment · security-awareness demonstrations.
