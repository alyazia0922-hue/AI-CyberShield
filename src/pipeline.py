"""Glue code: dataset -> clean -> detect -> score -> recommend, plus training CLI."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .anomaly_detection import MODEL_PATH, ModelBundle, load_model, predict, save_model, train_model
from .data_generator import generate_dataset
from .evaluation import evaluate
from .preprocessing import clean_dataframe
from .recommendations import generate_recommendations
from .risk_scoring import RiskConfig, overall_risk, score_events

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_PATH = DATA_DIR / "sample_dataset.csv"


def analyze(df_clean: pd.DataFrame, bundle: ModelBundle, cfg: RiskConfig | None = None) -> dict:
    """Run the full analysis chain on an already-cleaned dataframe."""
    cfg = cfg or RiskConfig()
    scored = score_events(predict(bundle, df_clean), cfg)
    overall = overall_risk(scored, cfg)
    recs = generate_recommendations(scored, overall)
    return {"scored": scored, "overall": overall, "recommendations": recs, "evaluation": evaluate(scored)}


def train_and_save(
    df_clean: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    path: Path | str = MODEL_PATH,
    seed: int = 42,
) -> tuple[ModelBundle, dict]:
    """Train on an unlabeled training split; evaluate on a held-out split.

    The model never sees ``is_attack`` - labels are only used for evaluation.
    Returns (bundle, test_metrics)."""
    stratify = None
    if "is_attack" in df_clean.columns and df_clean["is_attack"].nunique() == 2:
        stratify = df_clean["is_attack"]
    train_df, test_df = train_test_split(df_clean, test_size=0.3, random_state=seed, stratify=stratify)
    bundle = train_model(train_df, contamination=contamination, n_estimators=n_estimators, random_state=seed)
    metrics = evaluate(predict(bundle, test_df))
    if metrics.get("available"):
        bundle.meta["test_metrics"] = {k: metrics[k] for k in ("precision", "recall", "f1", "roc_auc", "n", "n_attacks")}
    bundle.meta["n_test_rows"] = int(len(test_df))
    save_model(bundle, path)
    return bundle, metrics


def ensure_sample_and_model() -> tuple[pd.DataFrame, ModelBundle]:
    """Create the sample dataset + trained model on first run."""
    DATA_DIR.mkdir(exist_ok=True)
    if SAMPLE_PATH.exists():
        raw = pd.read_csv(SAMPLE_PATH)
    else:
        raw = generate_dataset()
        raw.to_csv(SAMPLE_PATH, index=False)
    df, _ = clean_dataframe(raw)
    try:
        bundle = load_model()
    except Exception:
        bundle, _ = train_and_save(df)
    return df, bundle


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    raw = generate_dataset()
    raw.to_csv(SAMPLE_PATH, index=False)
    df, rep = clean_dataframe(raw)
    bundle, m = train_and_save(df)
    print(f"Sample dataset: {SAMPLE_PATH} ({len(df)} rows)")
    print(f"Model saved   : {MODEL_PATH}")
    if m.get("available"):
        print(f"Held-out test -> precision {m['precision']:.3f} recall {m['recall']:.3f} "
              f"F1 {m['f1']:.3f} ROC-AUC {m['roc_auc']:.3f}")
    res = analyze(df, bundle)
    print(f"Overall risk  : {res['overall']['score']} ({res['overall']['level']})")
