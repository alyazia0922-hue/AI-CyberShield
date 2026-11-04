"""Model evaluation when ground-truth labels (is_attack) exist."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def has_labels(df: pd.DataFrame) -> bool:
    return "is_attack" in df.columns and df["is_attack"].nunique() == 2


def evaluate(pred: pd.DataFrame) -> dict:
    """Compute metrics treating Suspicious+High as 'positive'."""
    if not has_labels(pred):
        return {"available": False, "reason": "The data has no 'is_attack' ground-truth label with both classes."}
    y = pred["is_attack"].astype(int).to_numpy()
    flagged = (pred["classification"].astype(str) != "Normal").astype(int).to_numpy()
    high = (pred["classification"].astype(str) == "High Risk").astype(int).to_numpy()
    score = pred["anomaly_score"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, score)

    per_type = None
    if "attack_type" in pred.columns:
        att = pred[pred["is_attack"] == 1]
        if len(att):
            g = att.groupby("attack_type", observed=True)["classification"].apply(
                lambda s: float((s.astype(str) != "Normal").mean())
            )
            per_type = g.rename("detection_rate").reset_index()

    return {
        "available": True,
        "precision": float(precision_score(y, flagged, zero_division=0)),
        "recall": float(recall_score(y, flagged, zero_division=0)),
        "f1": float(f1_score(y, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "avg_precision": float(average_precision_score(y, score)),
        "high_risk_precision": float(precision_score(y, high, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "confusion": np.array([[tn, fp], [fn, tp]]),
        "roc": (fpr, tpr),
        "per_type": per_type,
        "n": int(len(y)),
        "n_attacks": int(y.sum()),
    }
