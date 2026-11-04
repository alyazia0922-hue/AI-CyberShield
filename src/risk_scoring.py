"""Configurable 0-100 cyber risk score.

Two levels:
  * per-event risk  -> a weighted blend of simple, explainable factors
  * overall risk    -> how widespread each risk factor is across the dataset

Every weight and threshold lives in ``RiskConfig`` so the logic can be tuned
(from the Risk Assessment page) without touching code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .anomaly_detection import LABEL_HIGH, LABEL_SUSPICIOUS

FACTORS = ["failed_logins", "unusual_access", "network_traffic", "data_transfer", "anomaly_frequency"]

FACTOR_LABELS = {
    "failed_logins": "Failed logins",
    "unusual_access": "Unusual access patterns",
    "network_traffic": "Network traffic anomalies",
    "data_transfer": "Data transfer abnormalities",
    "anomaly_frequency": "Suspicious activity frequency",
}

FACTOR_HELP = {
    "failed_logins": "Share of records with many failed logins (>= threshold).",
    "unusual_access": "Share of records at off-hours (22:00-05:59) or from a new location.",
    "network_traffic": "Share of records with abnormal connection counts or port diversity.",
    "data_transfer": "Share of records with abnormally large outbound/inbound data.",
    "anomaly_frequency": "Share of records the ML model flagged (High Risk counts double).",
}


@dataclass
class RiskConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "failed_logins": 0.25,
            "unusual_access": 0.15,
            "network_traffic": 0.20,
            "data_transfer": 0.20,
            "anomaly_frequency": 0.20,
        }
    )
    # share of records at which a factor is considered "fully saturated" (=100)
    saturation: dict[str, float] = field(
        default_factory=lambda: {
            "failed_logins": 0.06,
            "unusual_access": 0.10,
            "network_traffic": 0.06,
            "data_transfer": 0.05,
            "anomaly_frequency": 0.12,
        }
    )
    failed_login_threshold: int = 5
    z_threshold: float = 4.0          # robust z for traffic / data abnormalities
    level_bounds: tuple[int, int, int] = (30, 60, 80)   # Low<=30, Medium<=60, High<=80, else Critical

    def normalized_weights(self) -> dict[str, float]:
        total = sum(max(w, 0) for w in self.weights.values())
        if total <= 0:
            return {k: 1 / len(FACTORS) for k in FACTORS}
        return {k: max(self.weights.get(k, 0), 0) / total for k in FACTORS}

    def to_dict(self) -> dict:
        return asdict(self)


def risk_level(score: float, bounds: tuple[int, int, int] = (30, 60, 80)) -> str:
    s = round(float(score))
    if s <= bounds[0]:
        return "Low"
    if s <= bounds[1]:
        return "Medium"
    if s <= bounds[2]:
        return "High"
    return "Critical"


LEVEL_ORDER = ["Low", "Medium", "High", "Critical"]


def factor_flags(pred: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    """Boolean per-record flags for each risk factor."""
    flags = pd.DataFrame(index=pred.index)
    flags["failed_logins"] = pred["failed_logins"] >= cfg.failed_login_threshold
    flags["unusual_access"] = (pred["off_hours"] == 1) | (pred["new_location"] == 1)
    flags["network_traffic"] = (pred["z_connections"] > cfg.z_threshold) | (pred["z_unique_ports"] > cfg.z_threshold)
    flags["data_transfer"] = (pred["z_log_bytes_out"] > cfg.z_threshold) | (pred["z_log_bytes_in"] > cfg.z_threshold)
    flags["anomaly_frequency"] = pred["classification"].astype(str) != "Normal"
    return flags


def score_events(pred: pd.DataFrame, cfg: RiskConfig | None = None) -> pd.DataFrame:
    """Add ``event_risk`` (0-100) and ``risk_level`` to every record."""
    cfg = cfg or RiskConfig()
    w = cfg.normalized_weights()
    comp = pd.DataFrame(index=pred.index)
    comp["failed_logins"] = np.clip(pred["failed_logins"] / (2 * cfg.failed_login_threshold), 0, 1)
    comp["unusual_access"] = np.clip(0.6 * pred["off_hours"] + 0.6 * pred["new_location"], 0, 1)
    comp["network_traffic"] = np.clip(
        np.maximum(pred["z_connections"], pred["z_unique_ports"]) / (3 * cfg.z_threshold), 0, 1
    )
    comp["data_transfer"] = np.clip(
        np.maximum(pred["z_log_bytes_out"], pred["z_log_bytes_in"]) / (3 * cfg.z_threshold), 0, 1
    )
    comp["anomaly_frequency"] = pred["anomaly_score"] / 100.0
    risk = sum(w[k] * comp[k] for k in FACTORS) * 100
    # records the model flags should never look "Low" just because few factors fire
    risk = np.where(pred["classification"].astype(str) == LABEL_HIGH, np.maximum(risk, 61), risk)
    risk = np.where(pred["classification"].astype(str) == LABEL_SUSPICIOUS, np.maximum(risk, 31), risk)

    out = pred.copy()
    out["event_risk"] = np.round(np.clip(risk, 0, 100), 1)
    out["risk_level"] = pd.Categorical(
        [risk_level(r, cfg.level_bounds) for r in out["event_risk"]], categories=LEVEL_ORDER, ordered=True
    )
    return out


def overall_risk(pred: pd.DataFrame, cfg: RiskConfig | None = None) -> dict:
    """Aggregate dataset-level score with a transparent factor breakdown."""
    cfg = cfg or RiskConfig()
    n = max(len(pred), 1)
    flags = factor_flags(pred, cfg)
    w = cfg.normalized_weights()

    shares = {k: float(flags[k].mean()) for k in FACTORS}
    cls = pred["classification"].astype(str)
    shares["anomaly_frequency"] = float(((cls == LABEL_SUSPICIOUS).sum() + 2 * (cls == LABEL_HIGH).sum()) / n)

    rows = []
    total = 0.0
    for k in FACTORS:
        sub = float(np.clip(shares[k] / max(cfg.saturation[k], 1e-9), 0, 1) * 100)
        contrib = w[k] * sub
        total += contrib
        rows.append(
            {
                "factor": FACTOR_LABELS[k],
                "key": k,
                "share_of_records_%": round(shares[k] * 100, 2),
                "sub_score": round(sub, 1),
                "weight_%": round(w[k] * 100, 1),
                "contribution": round(contrib, 1),
            }
        )
    total = float(np.clip(total, 0, 100))
    return {
        "score": round(total, 1),
        "level": risk_level(total, cfg.level_bounds),
        "breakdown": pd.DataFrame(rows),
        "n_records": int(n),
    }
