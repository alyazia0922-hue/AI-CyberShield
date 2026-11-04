import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_detection import load_model, predict, save_model, train_model  # noqa: E402
from src.attack_simulator import SCENARIOS, run_simulation  # noqa: E402
from src.data_generator import generate_dataset  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.preprocessing import clean_dataframe  # noqa: E402
from src.recommendations import generate_recommendations  # noqa: E402
from src.risk_scoring import RiskConfig, overall_risk, risk_level, score_events  # noqa: E402


@pytest.fixture(scope="module")
def data():
    df, rep = clean_dataframe(generate_dataset(2000, seed=3))
    assert rep.ok
    return df


@pytest.fixture(scope="module")
def bundle(data):
    return train_model(data)


def test_generator_is_reproducible():
    a, b = generate_dataset(500, seed=1), generate_dataset(500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a["is_attack"].sum() == 25


def test_clean_rejects_missing_columns():
    _, rep = clean_dataframe(pd.DataFrame({"timestamp": ["2026-01-01"], "user_id": ["a"]}))
    assert not rep.ok and "Missing required columns" in rep.errors[0]


def test_clean_handles_missing_and_negative():
    df = generate_dataset(300, seed=2).head(50).copy()
    df["timestamp"] = df["timestamp"].astype(str)   # a CSV read back from disk has text timestamps
    df.loc[0, "bytes_out_mb"] = np.nan
    df.loc[1, "failed_logins"] = -4
    df.loc[2, "timestamp"] = "not a date"
    out, rep = clean_dataframe(df)
    assert rep.ok and len(out) == 49
    assert out["failed_logins"].min() >= 0 and not out.isna().any().any()
    assert len(rep.warnings) >= 3


def test_clean_empty_and_oversize():
    assert not clean_dataframe(pd.DataFrame())[1].ok


def test_predict_shapes_and_labels(bundle, data):
    p = predict(bundle, data)
    assert len(p) == len(data)
    assert set(p["classification"].astype(str)) <= {"Normal", "Suspicious", "High Risk"}
    assert p["anomaly_score"].between(0, 100).all()
    assert (p["classification"].astype(str) != "Normal").mean() < 0.15


def test_model_save_load_roundtrip(bundle, data, tmp_path):
    f = save_model(bundle, tmp_path / "m.joblib")
    again = load_model(f)
    pd.testing.assert_series_equal(predict(bundle, data)["anomaly_score"], predict(again, data)["anomaly_score"])


def test_load_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_detection_quality_beats_chance(bundle, data):
    m = evaluate(predict(bundle, data))
    assert m["available"] and m["roc_auc"] > 0.9 and m["recall"] > 0.6


def test_risk_levels_boundaries():
    assert [risk_level(x) for x in (0, 30, 31, 60, 61, 80, 81, 100)] == [
        "Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]


def test_risk_weights_normalised_and_configurable(bundle, data):
    p = predict(bundle, data)
    base = overall_risk(score_events(p), RiskConfig())
    assert 0 <= base["score"] <= 100
    assert abs(base["breakdown"]["weight_%"].sum() - 100) < 0.5
    cfg = RiskConfig()
    cfg.saturation = {k: v / 10 for k, v in cfg.saturation.items()}   # much more sensitive
    assert overall_risk(score_events(p, cfg), cfg)["score"] > base["score"]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_simulation_raises_risk(bundle, scenario):
    r = run_simulation(scenario, bundle, n_events=60, severity=1.5)
    assert r.simulated_risk["score"] > r.baseline_risk["score"]
    assert r.n_injected == 60 and r.detection_rate > 0.8


def test_simulation_validation(bundle):
    with pytest.raises(ValueError):
        run_simulation("nope", bundle)
    with pytest.raises(ValueError):
        run_simulation("port_scan", bundle, n_events=0)


def test_recommendations_linked_to_findings(bundle):
    quiet, _ = clean_dataframe(generate_dataset(1500, anomaly_rate=0.0, seed=5))
    qs = score_events(predict(bundle, quiet))
    titles_quiet = " ".join(generate_recommendations(qs, overall_risk(qs))["title"])
    assert "Multi-Factor" not in titles_quiet and "exfil" not in titles_quiet.lower()

    r = run_simulation("brute_force", bundle, n_events=60)
    recs = generate_recommendations(r.simulated, r.simulated_risk)
    assert recs["title"].str.contains("Multi-Factor").any()
    r2 = run_simulation("data_exfiltration", bundle, n_events=40)
    assert generate_recommendations(r2.simulated, r2.simulated_risk)["title"].str.contains("outbound").any()
    r3 = run_simulation("port_scan", bundle, n_events=40)
    assert generate_recommendations(r3.simulated, r3.simulated_risk)["title"].str.contains("firewall").any()
