"""Synthetic security-event generator.

Everything here is fabricated with NumPy random draws. No real organisation
data is used and no network access is performed.

Each row represents one *activity record* (a user session / aggregated
connection window) with the columns documented in ``preprocessing.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATTACK_TYPES = [
    "brute_force",
    "port_scan",
    "suspicious_login",
    "data_exfiltration",
    "unusual_traffic",
]


def _make_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a small population of users with personal 'normal' profiles."""
    return pd.DataFrame(
        {
            "user_id": [f"user{i:03d}" for i in range(1, n_users + 1)],
            "ip": [f"10.0.{rng.integers(0, 8)}.{rng.integers(2, 250)}" for _ in range(n_users)],
            "out_scale": rng.lognormal(mean=1.2, sigma=0.4, size=n_users),  # MB
            "in_scale": rng.lognormal(mean=2.0, sigma=0.4, size=n_users),   # MB
            "conn_scale": rng.uniform(8, 30, size=n_users),
        }
    )


def generate_normal_events(
    n_events: int = 3000,
    n_users: int = 40,
    days: int = 14,
    start: str = "2026-10-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate benign business-like activity (mostly office hours)."""
    rng = np.random.default_rng(seed)
    users = _make_users(n_users, rng)
    u = users.iloc[rng.integers(0, n_users, size=n_events)].reset_index(drop=True)

    day = rng.integers(0, days, size=n_events)
    # office-hours activity around 13:00, small tail into evening
    hour = np.clip(rng.normal(13, 3.2, size=n_events), 7, 21.99)
    minute = rng.uniform(0, 60, size=n_events)
    ts = (
        pd.Timestamp(start)
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(np.floor(hour), unit="h")
        + pd.to_timedelta(np.floor(minute), unit="m")
    )

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "user_id": u["user_id"],
            "source_ip": u["ip"],
            "failed_logins": rng.poisson(0.25, size=n_events),
            "bytes_out_mb": u["out_scale"] * rng.lognormal(0, 0.45, size=n_events),
            "bytes_in_mb": u["in_scale"] * rng.lognormal(0, 0.45, size=n_events),
            "connections": np.maximum(1, rng.normal(u["conn_scale"], 4)).round().astype(int),
            "unique_ports": np.clip(rng.poisson(3, size=n_events) + 1, 1, 12),
            "session_minutes": np.clip(rng.normal(35, 15, size=n_events), 2, 120).round(1),
            "new_location": (rng.random(n_events) < 0.015).astype(int),
            "is_attack": 0,
            "attack_type": "none",
        }
    )
    return df


def _attack_events(
    kind: str,
    n: int,
    rng: np.random.Generator,
    users: list[str],
    start: pd.Timestamp,
    days: int,
    severity: float = 1.0,
    off_hours: bool = True,
) -> pd.DataFrame:
    """Create ``n`` malicious-looking (but purely synthetic) records."""
    if n <= 0:
        return pd.DataFrame()
    day = rng.integers(0, days, size=n)
    if off_hours:
        hour = rng.choice([0, 1, 2, 3, 4, 5, 23], size=n)
    else:
        hour = rng.integers(8, 20, size=n)
    minute = rng.integers(0, 60, size=n)
    ts = (
        start
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
    )
    victim = rng.choice(users, size=n)
    ip = [f"203.0.113.{rng.integers(2, 250)}" for _ in range(n)]  # TEST-NET-3 documentation range

    base = {
        "timestamp": ts,
        "user_id": victim,
        "source_ip": ip,
        "failed_logins": rng.poisson(0.25, size=n),
        "bytes_out_mb": rng.lognormal(1.2, 0.4, size=n),
        "bytes_in_mb": rng.lognormal(2.0, 0.4, size=n),
        "connections": rng.integers(8, 25, size=n),
        "unique_ports": rng.integers(1, 6, size=n),
        "session_minutes": rng.uniform(5, 45, size=n).round(1),
        "new_location": np.zeros(n, dtype=int),
    }

    if kind == "brute_force":
        base["failed_logins"] = rng.integers(int(15 * severity), int(60 * severity) + 2, size=n)
        base["connections"] = rng.integers(int(40 * severity), int(150 * severity) + 2, size=n)
        base["session_minutes"] = rng.uniform(1, 8, size=n).round(1)
        base["bytes_out_mb"] = rng.lognormal(0.2, 0.3, size=n)
        base["bytes_in_mb"] = rng.lognormal(0.5, 0.3, size=n)
    elif kind == "port_scan":
        base["unique_ports"] = rng.integers(int(60 * severity), int(400 * severity) + 2, size=n)
        base["connections"] = rng.integers(int(150 * severity), int(600 * severity) + 2, size=n)
        base["session_minutes"] = rng.uniform(1, 10, size=n).round(1)
        base["bytes_out_mb"] = rng.lognormal(-0.5, 0.3, size=n)
        base["bytes_in_mb"] = rng.lognormal(-0.3, 0.3, size=n)
    elif kind == "suspicious_login":
        base["new_location"] = np.ones(n, dtype=int)
        base["failed_logins"] = rng.integers(2, int(8 * severity) + 3, size=n)
        base["bytes_out_mb"] = rng.lognormal(1.2 + 0.8 * severity, 0.4, size=n)
        base["session_minutes"] = rng.uniform(20, 90, size=n).round(1)
    elif kind == "data_exfiltration":
        base["bytes_out_mb"] = rng.uniform(150 * severity, 900 * severity + 1, size=n)
        base["session_minutes"] = rng.uniform(60, 200, size=n).round(1)
        base["connections"] = rng.integers(20, 70, size=n)
    elif kind == "unusual_traffic":
        base["connections"] = rng.integers(int(200 * severity), int(900 * severity) + 2, size=n)
        base["bytes_in_mb"] = rng.uniform(80 * severity, 400 * severity + 1, size=n)
        base["bytes_out_mb"] = rng.uniform(20 * severity, 120 * severity + 1, size=n)
    else:
        raise ValueError(f"Unknown attack type: {kind}")

    out = pd.DataFrame(base)
    out["is_attack"] = 1
    out["attack_type"] = kind
    return out


def generate_dataset(
    n_events: int = 3000,
    anomaly_rate: float = 0.05,
    n_users: int = 40,
    days: int = 14,
    start: str = "2026-10-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Normal traffic plus a labelled share of mixed synthetic anomalies."""
    if not 0 <= anomaly_rate < 0.5:
        raise ValueError("anomaly_rate must be in [0, 0.5)")
    if n_events < 100:
        raise ValueError("n_events must be at least 100")

    rng = np.random.default_rng(seed)
    n_bad = int(round(n_events * anomaly_rate))
    normal = generate_normal_events(n_events - n_bad, n_users, days, start, seed)
    users = sorted(normal["user_id"].unique().tolist())
    start_ts = pd.Timestamp(start)

    # Hard benign cases (look unusual but are legitimate): big backups, night-shift
    # staff, travelling users. They make the detector produce realistic false positives.
    n_hard = int(len(normal) * 0.025)
    if n_hard:
        idx = rng.choice(normal.index.to_numpy(), size=n_hard, replace=False)
        third = n_hard // 3
        normal.loc[idx[:third], "bytes_out_mb"] *= rng.uniform(8, 25, size=third)       # backups
        night = idx[third: 2 * third]
        normal.loc[night, "timestamp"] = normal.loc[night, "timestamp"].dt.normalize() + pd.to_timedelta(
            rng.integers(0, 5, size=len(night)), unit="h")                              # night shift
        normal.loc[idx[2 * third:], "new_location"] = 1                                  # travel
        normal.loc[idx[2 * third:], "failed_logins"] += rng.integers(0, 3, size=len(idx) - 2 * third)

    parts = [normal]
    if n_bad:
        per_type = np.full(len(ATTACK_TYPES), n_bad // len(ATTACK_TYPES))
        per_type[: n_bad % len(ATTACK_TYPES)] += 1
        for kind, k in zip(ATTACK_TYPES, per_type):
            # 70% off-hours / 30% business hours, and 40% "low-and-slow" stealth variants
            # whose volume is only slightly above normal -> some attacks will be missed.
            k_off = int(round(k * 0.7))
            for n_part, off_flag in ((k_off, True), (k - k_off, False)):
                n_stealth = int(round(n_part * 0.4))
                parts.append(_attack_events(kind, n_part - n_stealth, rng, users, start_ts, days, 1.0, off_flag))
                parts.append(_attack_events(kind, n_stealth, rng, users, start_ts, days, 0.12, off_flag))

    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).sort_values("timestamp").reset_index(drop=True)
    df["bytes_out_mb"] = df["bytes_out_mb"].round(3)
    df["bytes_in_mb"] = df["bytes_in_mb"].round(3)
    return df
