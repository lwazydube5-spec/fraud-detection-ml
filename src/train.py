"""
src/train.py — Production Training Pipeline
=============================================
Builds a full sklearn Pipeline:

    FraudFeatureEngineer  →  StandardScaler  →  XGBClassifier

Steps:
  1. Load raw data
  2. Run 5-fold cross-validation to get honest OOF metrics
  3. Tune decision threshold on OOF predictions (maximises F1)
  4. Fit final model on full dataset
  5. Save pipeline + metadata to models/

XGBoost was chosen after comparing all three models in model_selection.py:
  - XGBoost      ROC-AUC 0.820  AP 0.204  Recall 67.6%
  - RandomForest ROC-AUC 0.808  AP 0.182  Recall 65.9%
  - LogisticReg  ROC-AUC 0.794  AP 0.160  Recall 54.1%

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

# ── XGBoost ────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

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

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1 on the fraud class.
    Uses OOF predictions so the threshold is never tuned on training data.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls),
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


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

        ("model",    XGBClassifier(
            # Tree structure
            n_estimators      = 400,
            max_depth         = 6,       # shallower than RF — XGB prefers this
            # Learning rate
            learning_rate     = 0.05,    # small steps → better generalisation
            # Row and column subsampling per tree
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            # Class imbalance — weights fraud cases 16× heavier than legit
            scale_pos_weight  = fraud_ratio,
            # Optimise directly for precision-recall AUC during training
            eval_metric       = "aucpr",
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbosity         = 0,
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
    threshold = tune_threshold(y.values, oof_probs)
    oof_preds = (oof_probs >= threshold).astype(int)

    print(f"  Best threshold : {threshold:.4f}")
    print_metrics(y, oof_preds, oof_probs, "Cross-Validated OOF Performance")

    # ── 5. Final fit on full dataset ────────────────────────────────────────
    # Retrain on ALL data — the production model should see everything.
    print("\n► Fitting final model on full dataset...")
    pipeline.fit(X_raw, y)

    # Feature importances
    xgb           = pipeline.named_steps["model"]
    feature_names = FraudFeatureEngineer().transform(X_raw).columns.tolist()
    importances   = pd.Series(xgb.feature_importances_, index=feature_names)
    top_features  = importances.nlargest(15)

    print("\nTop 15 features by importance:")
    for feat, imp in top_features.items():
        bar = "█" * int(imp * 300)
        print(f"  {feat:<35s} {imp:.4f}  {bar}")

    # ── 6. Save artefacts ───────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "model_type"       : "XGBClassifier",
        "threshold"        : threshold,
        "feature_names"    : feature_names,
        "top_features"     : top_features.to_dict(),
        "cv_roc_auc"       : round(float(roc_auc_score(y, oof_probs)), 4),
        "cv_avg_precision" : round(float(average_precision_score(y, oof_probs)), 4),
        "training_rows"    : len(X_raw),
        "fraud_rate"       : round(float(y.mean()), 4),
        "cv_folds"         : CV_FOLDS,
        "xgb_params"       : {
            "n_estimators"    : 400,
            "max_depth"       : 6,
            "learning_rate"   : 0.05,
            "subsample"       : 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": fraud_ratio,
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Pipeline saved  → {MODEL_PATH}")
    print(f"✓ Metadata saved  → {META_PATH}")
    print(f"\n  ROC-AUC (CV)       : {meta['cv_roc_auc']}")
    print(f"  Avg Precision (CV) : {meta['cv_avg_precision']}")
    print(f"  Threshold          : {threshold:.4f}")

    return pipeline, meta


if __name__ == "__main__":
    train()
