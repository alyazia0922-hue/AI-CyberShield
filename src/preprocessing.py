"""Validation, cleaning and feature engineering for security activity data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "user_id",
    "failed_logins",
    "bytes_out_mb",
    "bytes_in_mb",
    "connections",
    "unique_ports",
    "session_minutes",
]
OPTIONAL_COLUMNS = {
    "source_ip": "unknown",
    "new_location": 0,
    "is_attack": None,     # ground-truth label (0/1) if you have one
    "attack_type": None,
}
NUMERIC_COLUMNS = [
    "failed_logins",
    "bytes_out_mb",
    "bytes_in_mb",
    "connections",
    "unique_ports",
    "session_minutes",
    "new_location",
]

# Features fed to the model (order matters; saved with the model).
FEATURES = [
    "failed_logins",
    "off_hours",
    "new_location",
    "log_bytes_out",
    "log_bytes_in",
    "connections",
    "unique_ports",
    "session_minutes",
]

COLUMN_DOC = pd.DataFrame(
    [
        ("timestamp", "required", "Date-time of the activity (e.g. 2026-10-03 02:14:00)"),
        ("user_id", "required", "Pseudonymous user / account identifier"),
        ("failed_logins", "required", "Failed login attempts in the record (integer >= 0)"),
        ("bytes_out_mb", "required", "Data sent out, in megabytes (>= 0)"),
        ("bytes_in_mb", "required", "Data received, in megabytes (>= 0)"),
        ("connections", "required", "Number of network connections opened (>= 0)"),
        ("unique_ports", "required", "Distinct destination ports contacted (>= 0)"),
        ("session_minutes", "required", "Session duration in minutes (>= 0)"),
        ("source_ip", "optional", "Source IP (documentation/private ranges recommended)"),
        ("new_location", "optional", "1 if login came from a never-before-seen location, else 0"),
        ("is_attack", "optional", "Ground truth 0/1; enables precision/recall evaluation"),
        ("attack_type", "optional", "Free-text label for the ground-truth attack type"),
    ],
    columns=["column", "status", "description"],
)

MAX_ROWS = 200_000


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a raw dataframe.

    * checks required columns, row limits
    * parses timestamps (unparseable rows are dropped)
    * coerces numeric columns, fills missing values with the column median
    * clips negatives to zero and winsorises absurd outliers (> 1e6)
    * fills optional columns with safe defaults
    """
    rep = ValidationReport(rows_in=len(df))
    if df is None or len(df) == 0:
        rep.error("The file contains no rows.")
        return pd.DataFrame(), rep
    if len(df) > MAX_ROWS:
        rep.error(f"Too many rows ({len(df):,}); the limit is {MAX_ROWS:,}.")
        return pd.DataFrame(), rep

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rep.error("Missing required columns: " + ", ".join(missing))
        return pd.DataFrame(), rep

    # timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        rep.warnings.append(f"Dropped {bad_ts} row(s) with an unreadable timestamp.")
        df = df.dropna(subset=["timestamp"])
    if df.empty:
        rep.error("No rows with a valid timestamp remain.")
        return pd.DataFrame(), rep

    # user id
    df["user_id"] = df["user_id"].astype("string").fillna("unknown").astype(str)

    # optional columns
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns and default is not None:
            df[col] = default

    # numerics
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_na = int(df[col].isna().sum())
        if n_na:
            fill = 0.0 if col == "new_location" else float(df[col].median()) if n_na < len(df) else 0.0
            df[col] = df[col].fillna(fill)
            rep.warnings.append(f"Filled {n_na} missing value(s) in '{col}' with {fill:g}.")
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            df[col] = df[col].clip(lower=0)
            rep.warnings.append(f"Clipped {n_neg} negative value(s) in '{col}' to 0.")
        df[col] = df[col].clip(upper=1e6)
    df["new_location"] = (df["new_location"] > 0).astype(int)

    if "is_attack" in df.columns:
        lab = pd.to_numeric(df["is_attack"], errors="coerce")
        if lab.notna().all():
            df["is_attack"] = (lab > 0).astype(int)
        else:
            df = df.drop(columns=["is_attack"])
            rep.warnings.append("'is_attack' contained non-numeric values and was ignored.")

    dup = int(df.duplicated().sum())
    if dup:
        rep.warnings.append(f"{dup} exact duplicate row(s) kept (they may be legitimate repeated events).")

    df = df.sort_values("timestamp").reset_index(drop=True)
    rep.rows_out = len(df)
    return df, rep


def is_off_hours(hour: pd.Series | np.ndarray) -> np.ndarray:
    """Off-hours = 22:00-05:59 local time."""
    h = np.asarray(hour)
    return ((h >= 22) | (h < 6)).astype(int)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived columns used by the model and risk engine."""
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["off_hours"] = is_off_hours(out["hour"])
    out["log_bytes_out"] = np.log1p(out["bytes_out_mb"])
    out["log_bytes_in"] = np.log1p(out["bytes_in_mb"])
    return out


def feature_matrix(df_features: pd.DataFrame) -> pd.DataFrame:
    return df_features[FEATURES].astype(float)
