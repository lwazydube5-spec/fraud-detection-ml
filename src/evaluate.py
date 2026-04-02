"""
evaluate.py — Model Evaluation & Reporting
==========================================
Loads the saved model, runs a proper train/test split evaluation,
and produces a comprehensive performance report.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent))
from features import load_raw

DATA_PATH  = Path(__file__).parent.parent / "data" / "fraud_data.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "fraud_model.pkl"
META_PATH  = Path(__file__).parent.parent / "models" / "model_meta.json"
REPORT_DIR = Path(__file__).parent.parent / "models"

RANDOM_STATE = 42
TEST_SIZE    = 0.2


def evaluate_holdout():
    print("► Loading model and metadata …")
    pipeline  = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    threshold = meta["threshold"]

    print("► Loading data and creating holdout split …")
    X_raw, y = load_raw(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE,
    )

    # Re-fit on train split only for honest holdout evaluation
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics ─────────────────────────────────────────────────────────
    roc_auc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    cm       = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = {
        "threshold":            threshold,
        "test_size":            len(y_test),
        "fraud_cases_in_test":  int(y_test.sum()),
        "roc_auc":              round(roc_auc, 4),
        "average_precision":    round(avg_prec, 4),
        "true_positives":       int(tp),
        "false_positives":      int(fp),
        "true_negatives":       int(tn),
        "false_negatives":      int(fn),
        "precision_fraud":      round(tp / (tp + fp) if tp + fp > 0 else 0, 4),
        "recall_fraud":         round(tp / (tp + fn) if tp + fn > 0 else 0, 4),
        "f1_fraud":             round(
            2 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) > 0 else 0, 4
        ),
    }

    # ── Print report ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HOLDOUT TEST SET EVALUATION")
    print("="*60)
    print(f"  Test rows         : {report['test_size']:,}")
    print(f"  Fraud cases       : {report['fraud_cases_in_test']} ({report['fraud_cases_in_test']/report['test_size']:.1%})")
    print(f"  Decision threshold: {threshold:.4f}")
    print()
    print(f"  ROC-AUC           : {report['roc_auc']:.4f}")
    print(f"  Avg Precision     : {report['average_precision']:.4f}")
    print()
    print("  Fraud class metrics:")
    print(f"    Precision : {report['precision_fraud']:.4f}")
    print(f"    Recall    : {report['recall_fraud']:.4f}")
    print(f"    F1-Score  : {report['f1_fraud']:.4f}")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Pred Legit  Pred Fraud")
    print(f"  Actual Legit   {tn:6d}      {fp:6d}")
    print(f"  Actual Fraud   {fn:6d}      {tp:6d}")
    print()

    business_impact(tn, fp, fn, tp)

    # ── Full sklearn report ───────────────────────────────────────────
    print("\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Save report
    report_path = REPORT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Evaluation report saved → {report_path}")

    return report


def business_impact(tn, fp, fn, tp, avg_fraud_amount=15000, investigation_cost=200):
    """
    Translate ML metrics into business dollar impact.
    Assumptions (adjust to your context):
      - Average fraudulent claim: $15,000
      - Cost to investigate a flagged claim: $200
    """
    caught_fraud_value     = tp * avg_fraud_amount
    missed_fraud_value     = fn * avg_fraud_amount
    investigation_costs    = (tp + fp) * investigation_cost
    net_benefit            = caught_fraud_value - investigation_costs

    print("  Business Impact Estimate (per period):")
    print(f"    Fraud caught (${avg_fraud_amount:,}/claim) : ${caught_fraud_value:>12,.0f}")
    print(f"    Fraud missed                   : ${missed_fraud_value:>12,.0f}")
    print(f"    Investigation costs            : ${investigation_costs:>12,.0f}  ({tp+fp} claims @ ${investigation_cost})")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Net estimated benefit          : ${net_benefit:>12,.0f}")


if __name__ == "__main__":
    evaluate_holdout()
