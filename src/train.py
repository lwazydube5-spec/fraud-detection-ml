"""
src/train.py — Production Training Pipeline
=============================================
Builds a full sklearn Pipeline:

    FraudFeatureEngineer  →  StandardScaler  →  RandomForestClassifier

Steps:
  1. Load raw data
  2. Run 5-fold cross-validation to get honest OOF metrics
  3. Set threshold
  4. Fit final model on full dataset
  5. Save pipeline + metadata to models/

Random Forest was chosen after comparing all three models in model_selection.py:
  - Random Forest  ROC-AUC 0.806  Recall 95.6%  ← winner on recall
  - XGBoost        ROC-AUC 0.812  Recall 72.5%
  - LogisticReg    ROC-AUC 0.793  Recall 93.0%

Recall is the priority metric — missing fraud costs $15,000 vs $200 for a false alarm.
Threshold hardcoded at 0.30 based on cost-benefit analysis.

Usage:
    python src/train.py
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── sklearn ────────────────────────────────────────────────────────────────
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

# ── RandomForest ───────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

# ── Local ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from features import FraudFeatureEngineer, load_raw

# ─────────────────────────────── Config ────────────────────────────────────

DATA_PATH    = Path(__file__).parent.parent / "data"   / "fraud_data.csv"
MODEL_DIR    = Path(__file__).parent.parent / "models"
MODEL_PATH   = MODEL_DIR / "fraud_model.pkl"
META_PATH    = MODEL_DIR / "model_meta.json"

RANDOM_STATE = 42
CV_FOLDS     = 5


# ─────────────────────────────── Helpers ───────────────────────────────────

THRESHOLD = 0.30

def print_metrics(y_true, y_pred, y_prob, label=""):
    """Print a full evaluation block."""
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC       : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Avg Precision : {average_precision_score(y_true, y_prob):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion matrix:")
    print(f"              Pred Legit  Pred Fraud")
    print(f"  Actual Legit   {tn:6d}      {fp:6d}")
    print(f"  Actual Fraud   {fn:6d}      {tp:6d}")


# ─────────────────────────────── Main ──────────────────────────────────────

def train():
    print("=" * 58)
    print("  Fraud Detection — Production Training")
    print("=" * 58)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n► Loading data...")
    X_raw, y = load_raw(DATA_PATH)
    print(f"  Rows       : {len(X_raw):,}")
    print(f"  Fraud rate : {y.mean():.2%}  ({y.sum()} fraud / {len(y):,} total)")

    # scale_pos_weight tells XGBoost how much rarer fraud is than legit
    fraud_ratio = int((y == 0).sum() / (y == 1).sum())
    print(f"  Class ratio: {fraud_ratio}:1  (legit:fraud)")

    # ── 2. Build Pipeline ───────────────────────────────────────────────────
    # Three steps chained in order:
    #   FraudFeatureEngineer  raw strings → 96 numeric features
    #   StandardScaler        mean=0, std=1 for every column
    #   XGBClassifier         gradient boosted trees, imbalance-aware
    #
    # The Pipeline applies all three steps automatically in the right order
    # at both training time and inference time — no manual coordination needed.
    print("\n► Building Pipeline...")
    pipeline = Pipeline([
    ("features", FraudFeatureEngineer()),
    ("scaler",   StandardScaler()),
    ("model",    RandomForestClassifier(
        n_estimators     = 400,
        max_depth        = 12,
        min_samples_leaf = 5,
        class_weight     = "balanced",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )),
])


    print("  Steps:")
    for name, step in pipeline.steps:
        print(f"    {name:<12} → {step.__class__.__name__}")

    # ── 3. Cross-validated OOF predictions ─────────────────────────────────
    # cross_val_predict generates predictions for each sample only while
    # that sample was in the held-out fold — gives honest, unbiased metrics.
    print(f"\n► Running {CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(
        n_splits     = CV_FOLDS,
        shuffle      = True,
        random_state = RANDOM_STATE,
    )

    oof_probs = cross_val_predict(
        pipeline, X_raw, y,
        cv     = cv,
        method = "predict_proba",
        n_jobs = -1,
    )[:, 1]

    # ── 4. Threshold tuning ─────────────────────────────────────────────────
    # Default 0.5 threshold is wrong for imbalanced data.
    # We find the value that maximises F1 on the fraud class.
    oof_preds = (oof_probs >= THRESHOLD).astype(int)
    print(f"  Fixed threshold : {THRESHOLD}")
    print(f"  Fraud caught    : {oof_preds[y.values==1].sum()} / {y.sum()}")
    print(f"  False alarms    : {((oof_preds==1) & (y.values==0)).sum()}")

    print_metrics(y, oof_preds, oof_probs, "Cross-Validated OOF Performance")

    # ── 5. Final fit on full dataset ────────────────────────────────────────
    # Retrain on ALL data — the production model should see everything.
    print("\n► Fitting final model on full dataset...")
    pipeline.fit(X_raw, y)

    # Feature importances
    rf           = pipeline.named_steps["model"]
    feature_names = FraudFeatureEngineer().transform(X_raw).columns.tolist()
    importances   = pd.Series(rf.feature_importances_, index=feature_names)
    top_features  = importances.nlargest(15)

    print("\nTop 15 features by importance:")
    for feat, imp in top_features.items():
        bar = "█" * int(imp * 300)
        print(f"  {feat:<35s} {imp:.4f}  {bar}")

    # ── 6. Save artefacts ───────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
    "model_type"       : "RandomForestClassifier",
    "threshold"        : THRESHOLD,
    "feature_names"    : feature_names,
    "top_features"     : top_features.to_dict(),
    "cv_roc_auc"       : round(float(roc_auc_score(y, oof_probs)), 4),
    "cv_avg_precision" : round(float(average_precision_score(y, oof_probs)), 4),
    "cv_recall"        : round(float((oof_preds[y.values==1]).sum() / y.sum()), 4),
    "training_rows"    : len(X_raw),
    "fraud_rate"       : round(float(y.mean()), 4),
    "cv_folds"         : CV_FOLDS,
    "rf_params"        : {
        "n_estimators"    : 400,
        "max_depth"       : 12,
        "min_samples_leaf": 5,
        "class_weight"    : "balanced",
    },
}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Pipeline saved  → {MODEL_PATH}")
    print(f"✓ Metadata saved  → {META_PATH}")
    print(f"\n  ROC-AUC (CV)       : {meta['cv_roc_auc']}")
    print(f"  Avg Precision (CV) : {meta['cv_avg_precision']}")
    print(f"  Recall (CV)        : {meta['cv_recall']:.1%}")
    print(f"  Threshold          : {THRESHOLD}")

    return pipeline, meta


if __name__ == "__main__":
    train()
