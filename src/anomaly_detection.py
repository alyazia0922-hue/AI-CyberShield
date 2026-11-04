"""Isolation Forest anomaly detector (train / save / load / predict / explain).

Important: anomaly detection finds *unusual* behaviour. An unusual record is
not proof of an attack, and a normal-looking record is not proof of safety.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import FEATURES, add_features, feature_matrix

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"

LABEL_NORMAL = "Normal"
LABEL_SUSPICIOUS = "Suspicious"
LABEL_HIGH = "High Risk"
LABELS = [LABEL_NORMAL, LABEL_SUSPICIOUS, LABEL_HIGH]

# features used for human-readable explanations (robust z-score vs. normal)
EXPLAIN_FEATURES = {
    "failed_logins": "failed logins",
    "log_bytes_out": "outbound data volume",
    "log_bytes_in": "inbound data volume",
    "connections": "connection count",
    "unique_ports": "distinct ports contacted",
    "session_minutes": "session duration",
}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    features: list[str]
    score_ref_median: float      # median of (-score_samples) on training data
    score_ref_max: float         # 99.9th percentile of (-score_samples)
    high_cut: float              # decision_function below this => High Risk
    feature_median: dict[str, float]
    feature_mad: dict[str, float]
    params: dict
    meta: dict = field(default_factory=dict)


def train_model(
    df_raw_clean: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    max_samples: str | int = "auto",
    high_risk_quantile: float = 0.01,
    random_state: int = 42,
) -> ModelBundle:
    """Fit scaler + Isolation Forest on unlabeled activity data."""
    if len(df_raw_clean) < 50:
        raise ValueError("At least 50 rows are required to train the model.")
    if not 0.001 <= contamination <= 0.5:
        raise ValueError("contamination must be between 0.001 and 0.5")

    feats = add_features(df_raw_clean)
    X = feature_matrix(feats)

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X)

    Xs = pipe.named_steps["scale"].transform(X)
    forest = pipe.named_steps["iforest"]
    neg = -forest.score_samples(Xs)
    dec = forest.decision_function(Xs)

    med = {c: float(np.median(X[c])) for c in FEATURES}
    mad = {c: float(np.median(np.abs(X[c] - med[c]))) for c in FEATURES}

    return ModelBundle(
        pipeline=pipe,
        features=list(FEATURES),
        score_ref_median=float(np.median(neg)),
        score_ref_max=float(np.quantile(neg, 0.999)),
        high_cut=float(np.quantile(dec, high_risk_quantile)),
        feature_median=med,
        feature_mad=mad,
        params={
            "contamination": contamination,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "high_risk_quantile": high_risk_quantile,
            "random_state": random_state,
        },
        meta={
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_train_rows": int(len(df_raw_clean)),
        },
    )


def save_model(bundle: ModelBundle, path: Path | str = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: Path | str = MODEL_PATH) -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Train one first.")
    obj = joblib.load(path)
    if not isinstance(obj, ModelBundle):
        raise ValueError("File is not a valid AI CyberShield model bundle.")
    return obj


def _robust_z(values: pd.Series, med: float, mad: float) -> pd.Series:
    scale = max(1.4826 * mad, 1e-6)
    scale = max(scale, 0.05 * max(abs(med), 1.0))  # avoid exploding on near-constant features
    return (values - med) / scale


def predict(bundle: ModelBundle, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Score every record. Returns a dataframe with features + detection columns.

    New columns:
      anomaly_score (0-100, higher = more unusual), decision (raw IF value),
      classification (Normal/Suspicious/High Risk), reasons (text),
      z_* robust z-scores for the explainable features.
    """
    feats = add_features(df_clean)
    X = feature_matrix(feats)[bundle.features]
    Xs = bundle.pipeline.named_steps["scale"].transform(X)
    forest = bundle.pipeline.named_steps["iforest"]

    neg = -forest.score_samples(Xs)
    dec = forest.decision_function(Xs)
    denom = max(bundle.score_ref_max - bundle.score_ref_median, 1e-9)
    anomaly_score = np.clip((neg - bundle.score_ref_median) / denom, 0, 1) * 100

    cls = np.where(dec < bundle.high_cut, LABEL_HIGH, np.where(dec < 0, LABEL_SUSPICIOUS, LABEL_NORMAL))

    out = feats.copy()
    out["decision"] = dec
    out["anomaly_score"] = np.round(anomaly_score, 1)
    out["classification"] = pd.Categorical(cls, categories=LABELS, ordered=True)

    zcols = {}
    for f in EXPLAIN_FEATURES:
        zcols[f"z_{f}"] = _robust_z(out[f], bundle.feature_median[f], bundle.feature_mad[f])
    out = pd.concat([out, pd.DataFrame(zcols, index=out.index)], axis=1)
    out["reasons"] = _build_reasons(out)
    return out


def _build_reasons(df: pd.DataFrame, z_thr: float = 3.0) -> list[str]:
    reasons: list[list[str]] = [[] for _ in range(len(df))]
    for f, label in EXPLAIN_FEATURES.items():
        flagged = df[f"z_{f}"].to_numpy() > z_thr
        for i in np.flatnonzero(flagged):
            reasons[i].append(f"high {label}")
    off = df["off_hours"].to_numpy() == 1
    newloc = df["new_location"].to_numpy() == 1
    for i in np.flatnonzero(off):
        reasons[i].append("off-hours access")
    for i in np.flatnonzero(newloc):
        reasons[i].append("login from new location")
    flagged_cls = df["classification"].to_numpy() != LABEL_NORMAL
    return [
        ", ".join(r) if r else ("statistically unusual combination of features" if flagged_cls[i] else "")
        for i, r in enumerate(reasons)
    ]
