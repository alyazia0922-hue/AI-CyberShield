"""Attack simulation engine (100% synthetic, offline).

A scenario injects fabricated malicious-looking *records* into a benign
baseline dataset. Nothing is executed and no network connection is made: the
"attack" is just rows in a DataFrame that we then feed through the detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomaly_detection import ModelBundle, predict
from .data_generator import _attack_events, generate_normal_events
from .risk_scoring import RiskConfig, overall_risk, score_events

SCENARIOS: dict[str, dict] = {
    "brute_force": {
        "name": "Brute Force Attack",
        "description": "Many rapid failed logins against accounts, with high connection counts.",
        "sliders": {"events": (5, 150, 40, "Malicious records injected"),
                    "severity": (0.5, 3.0, 1.0, "Intensity (failed-login volume)")},
    },
    "port_scan": {
        "name": "Port Scanning",
        "description": "One source probes hundreds of ports with tiny, short connections.",
        "sliders": {"events": (5, 150, 30, "Malicious records injected"),
                    "severity": (0.5, 3.0, 1.0, "Intensity (ports probed)")},
    },
    "suspicious_login": {
        "name": "Suspicious Login",
        "description": "Logins from a never-seen location with a few failures and elevated data use.",
        "sliders": {"events": (5, 150, 25, "Malicious records injected"),
                    "severity": (0.5, 3.0, 1.0, "Intensity")},
    },
    "data_exfiltration": {
        "name": "Data Exfiltration",
        "description": "Very large, long outbound transfers, typically outside business hours.",
        "sliders": {"events": (5, 150, 20, "Malicious records injected"),
                    "severity": (0.5, 3.0, 1.0, "Intensity (data volume)")},
    },
    "unusual_traffic": {
        "name": "Unusual Network Traffic",
        "description": "Sudden surge of connections and traffic volume (DDoS / beaconing-like).",
        "sliders": {"events": (5, 150, 35, "Malicious records injected"),
                    "severity": (0.5, 3.0, 1.0, "Intensity (traffic volume)")},
    },
}


@dataclass
class SimulationResult:
    scenario: str
    baseline: pd.DataFrame     # scored baseline (no attack)
    simulated: pd.DataFrame    # scored baseline + injected attack
    baseline_risk: dict
    simulated_risk: dict
    n_injected: int
    n_detected: int            # injected records classified Suspicious/High Risk

    @property
    def detection_rate(self) -> float:
        return self.n_detected / self.n_injected if self.n_injected else float("nan")


def run_simulation(
    scenario: str,
    bundle: ModelBundle,
    n_events: int = 40,
    severity: float = 1.0,
    off_hours: bool = True,
    baseline_size: int = 1500,
    seed: int = 7,
    cfg: RiskConfig | None = None,
) -> SimulationResult:
    """Inject ``n_events`` synthetic malicious records and re-score."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}")
    if not 1 <= n_events <= 5000:
        raise ValueError("n_events must be between 1 and 5000")
    if not 0.1 <= severity <= 10:
        raise ValueError("severity must be between 0.1 and 10")
    cfg = cfg or RiskConfig()

    rng = np.random.default_rng(seed)
    base_clean = generate_normal_events(n_events=baseline_size, seed=seed)
    users = sorted(base_clean["user_id"].unique().tolist())
    attack = _attack_events(scenario, n_events, rng, users, pd.Timestamp("2026-10-01"), 14, severity, off_hours)
    sim_clean = pd.concat([base_clean, attack], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    base_scored = score_events(predict(bundle, base_clean), cfg)
    sim_scored = score_events(predict(bundle, sim_clean), cfg)

    inj = sim_scored[sim_scored["is_attack"] == 1]
    detected = int((inj["classification"].astype(str) != "Normal").sum())
    return SimulationResult(
        scenario=scenario,
        baseline=base_scored,
        simulated=sim_scored,
        baseline_risk=overall_risk(base_scored, cfg),
        simulated_risk=overall_risk(sim_scored, cfg),
        n_injected=int(len(inj)),
        n_detected=detected,
    )
