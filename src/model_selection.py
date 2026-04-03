"""
src/model_selection.py — Compare Models Before Committing to One
=================================================================
This script should be run BEFORE train.py.

We test three candidate models using cross-validation and pick the
best one based on Recall on the fraud (minority) class.

Models compared:
  1. Logistic Regression  — simple linear baseline
  2. Random Forest        — ensemble of independent trees
  3. XGBoost              — ensemble of boosted trees (sequential)

Usage:
    python src/model_selection.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, f1_score,
    precision_score, recall_score,precision_recall_curve,
)

# Try importing XGBoost — give a clear message if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠  XGBoost not installed. Run: pip install xgboost")
    print("   Continuing with Logistic Regression and Random Forest only.\n")

sys.path.insert(0, str(Path(__file__).parent))
from features import FraudFeatureEngineer, load_raw

# ─────────────────────────────── Config ────────────────────────────────────

DATA_PATH    = Path(__file__).parent.parent / "data" / "fraud_data.csv"
RANDOM_STATE = 42
CV_FOLDS     = 5


# ─────────────────────────────── Helpers ───────────────────────────────────

def evaluate_model(name, pipeline, X, y, cv):
    """
    Run cross-validated evaluation for one model.
    Returns a dict of metrics.
    """
    print(f"  Testing {name}...", end=" ", flush=True)
    t0 = perf_counter()

    # Out-of-fold probability predictions
    oof_probs = cross_val_predict(
        pipeline, X, y,
        cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    elapsed = perf_counter() - t0

    # Threshold on OOF probs
    threshold  = 0.3
    oof_preds  = (oof_probs >= threshold).astype(int)

    roc_auc    = roc_auc_score(y, oof_probs)
    avg_prec   = average_precision_score(y, oof_probs)
    f1         = f1_score(y, oof_preds, pos_label=1)
    precision  = precision_score(y, oof_preds, pos_label=1, zero_division=0)
    recall     = recall_score(y, oof_preds, pos_label=1)

    print(f"done ({elapsed:.0f}s)")

    return {
        "model":       name,
        "roc_auc":     round(roc_auc,   4),
        "avg_prec":    round(avg_prec,  4),
        "f1_fraud":    round(f1,        4),
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "threshold":   round(threshold, 4),
        "train_time":  round(elapsed,   1),
        "oof_probs":   oof_probs,
        "oof_preds":   oof_preds,
    }


def print_results_table(results):
    """Print a clean comparison table."""
    print()
    print("=" * 72)
    print(f"  {'Model':<25} {'ROC-AUC':>8} {'Avg Prec':>9} {'F1':>7} {'Recall':>8} {'Prec':>7}")
    print("=" * 72)
    for r in results:
        marker = "  ◀ best Recall" if r == results[0] else ""
        print(
            f"  {r['model']:<25} "
            f"{r['roc_auc']:>8.4f} "
            f"{r['avg_prec']:>9.4f} "
            f"{r['f1_fraud']:>7.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['precision']:>7.4f}"
            f"{marker}"
        )
    print("=" * 72)


def print_classification_reports(results, y):
    """Print full per-class report for each model."""
    for r in results:
        print(f"\n── {r['model']} ──────────────────────────────────────")
        print(f"   Threshold: {r['threshold']:.4f}  |  Train time: {r['train_time']}s")
        print(classification_report(
            y, r['oof_preds'],
            target_names=["Legit", "Fraud"],
            digits=3,
        ))


# ─────────────────────────────── Main ──────────────────────────────────────

def run_model_selection():
    print("=" * 55)
    print("  Model Selection — Fraud Detection")
    print("=" * 55)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n► Loading data...")
    X_raw, y = load_raw(DATA_PATH)
    print(f"  Rows: {len(X_raw):,}  |  Fraud rate: {y.mean():.2%}")

    # ── Pre-engineer features once (same for all models) ─────────────────
    # We transform upfront so we're not re-running feature engineering
    # inside every CV fold for every model — faster and fairer comparison
    print("\n► Engineering features...")
    eng    = FraudFeatureEngineer()
    X_eng  = eng.fit_transform(X_raw)
    print(f"  Features: {X_eng.shape[1]}")

    # ── Define candidate models ──────────────────────────────────────────
    # Each model is wrapped with a StandardScaler.
    # Logistic Regression needs scaling (it's sensitive to feature magnitude).
    # Tree models don't need it but it doesn't hurt them either.
    fraud_ratio = int((y == 0).sum() / (y == 1).sum())  # ~16

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                class_weight  = "balanced",  # handles imbalance
                max_iter      = 1000,        # enough iterations to converge
                C             = 0.1,         # regularisation — prevents overfit
                solver        = "lbfgs",
                random_state  = RANDOM_STATE,
                n_jobs        = -1,
            )),
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                n_estimators     = 300,
                max_depth        = 12,
                min_samples_leaf = 5,
                class_weight     = "balanced",
                random_state     = RANDOM_STATE,
                n_jobs           = -1,
            )),
        ]),
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(
                n_estimators      = 300,
                max_depth         = 6,
                learning_rate     = 0.05,    # small steps → better generalisation
                subsample         = 0.8,     # 80% of rows per tree
                colsample_bytree  = 0.8,     # 80% of features per tree
                scale_pos_weight  = fraud_ratio,  # handles imbalance (same idea as balanced)
                eval_metric       = "aucpr",
                random_state      = RANDOM_STATE,
                n_jobs            = -1,
                verbosity         = 0,
            )),
        ])

    # ── Cross-validate all models ────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n► Running {CV_FOLDS}-fold cross-validation on {len(candidates)} models...")
    results = []
    for name, pipeline in candidates.items():
        result = evaluate_model(name, pipeline, X_eng, y, cv)
        results.append(result)

    # Sort by ROC-AUC descending
    results.sort(key=lambda r: r["recall"], reverse=True)

    # ── Print comparison table ───────────────────────────────────────────
    print_results_table(results)

    # ── Print full classification reports ───────────────────────────────
    print("\n► Full classification reports (OOF predictions):")
    print_classification_reports(results, y)

    # ── Winner ──────────────────────────────────────────────────────────
    winner = results[0]
    print("\n" + "=" * 55)
    print(f"  Winner: {winner['model']}")
    print(f"  ROC-AUC  : {winner['roc_auc']}")
    print(f"  F1 fraud : {winner['f1_fraud']}")
    print(f"  Recall   : {winner['recall']}")
    print(f"  Threshold: {winner['threshold']}")
    print("=" * 55)
    print(f"\n  → Use {winner['model']} in train.py")
    print()

    return results


if __name__ == "__main__":
    run_model_selection()
