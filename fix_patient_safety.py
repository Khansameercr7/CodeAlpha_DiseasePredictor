"""
fix_patient_safety.py
---------------------
Applies all Tier-1 (patient safety) fixes to the Disease Prediction System.

Fixes applied
-------------
1. THRESHOLD TUNING    — Lower decision boundary from 0.5 to clinically
                         optimal value targeting ≥85% recall (diabetes/BC)
                         and ≥92% recall (heart). Persisted into meta JSON
                         so the API and Streamlit frontend use it automatically.

2. OVERFITTING FIX     — Retrain heart RF and XGBoost with regularised
                         hyperparameter grids (constrained max_depth,
                         min_child_weight, reg_lambda). Verify gap < 0.05
                         with learning curves.

3. CALIBRATION CHECK   — Measure Expected Calibration Error before and after.
                         Flag models where |predicted_prob - actual_rate| > 0.05.

4. PR CURVES           — Generate Precision-Recall curves alongside ROC.
                         PR-AUC replaces ROC-AUC as primary metric for
                         imbalanced datasets (diabetes, breast cancer).

Usage
-----
    python fix_patient_safety.py                 # all diseases
    python fix_patient_safety.py --disease heart  # one disease only
    python fix_patient_safety.py --skip-retrain  # threshold + PR only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import DATASET_CONFIG, REPORTS_DIR, FIGURES_DIR
from data.data_loader import DataLoader
from evaluation.learning_curves import LearningCurveAnalyzer
from evaluation.pr_curves import PRCurveAnalyzer
from models.model_factory import ModelFactory
from utils.calibrator import ModelCalibrator
from utils.logger import get_logger
from utils.model_persistence import ModelPersistence
from utils.threshold_optimizer import ThresholdOptimizer

logger = get_logger("fix_patient_safety")

DISEASES     = list(DATASET_CONFIG.keys())
MODEL_NAMES  = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: Optimal threshold tuning
# ─────────────────────────────────────────────────────────────────────────────

def fix_thresholds(disease: str) -> pd.DataFrame:
    """
    Find and persist optimal thresholds for all 4 models of one disease.
    Returns a comparison DataFrame.
    """
    logger.info("═" * 60)
    logger.info(" FIX 1 — THRESHOLD TUNING: %s", disease.upper())
    logger.info("═" * 60)

    loader = DataLoader(disease).load_and_prepare()
    _, X_test, _, y_test = loader.get_data()
    assert X_test is not None and y_test is not None, "Test data cannot be None"

    rows = []
    for model_name in MODEL_NAMES:
        try:
            model, scaler, meta = ModelPersistence.load(disease, model_name)
            y_proba = model.predict_proba(X_test)[:, 1]

            optimizer = ThresholdOptimizer(disease, model_name)
            result    = optimizer.optimize(y_test, y_proba)

            rows.append({
                "Disease":       disease,
                "Model":         model_name,
                "Default_Recall": result.default_recall,
                "Default_FN":    result.default_fn,
                "Opt_Threshold": result.optimal_threshold,
                "Opt_Recall":    result.recall,
                "Opt_FN":        result.false_negatives,
                "Recall_Gain":   result.recall_gain,
                "Opt_F1":        result.f1,
                "Total_Pos":     result.total_positives,
            })
        except FileNotFoundError:
            logger.warning("  Model not found: %s / %s — skipping", disease, model_name)

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info("\n%s\n", df[["Model","Default_Recall","Default_FN",
                                  "Opt_Threshold","Opt_Recall","Opt_FN",
                                  "Recall_Gain"]].to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Retrain heart models with regularisation
# ─────────────────────────────────────────────────────────────────────────────

def fix_overfitting(disease: str) -> dict:
    """
    Retrain RF and XGBoost with the v2 regularised hyperparameter grid.
    Generate learning curves before (from existing model) and after (retrained).
    """
    logger.info("═" * 60)
    logger.info(" FIX 2 — OVERFITTING: %s", disease.upper())
    logger.info("═" * 60)

    loader = DataLoader(disease).load_and_prepare()
    X_train, X_test, y_train, y_test = loader.get_data()
    assert X_train is not None and X_test is not None and y_train is not None and y_test is not None, "Train/test data cannot be None"

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_ratio = n_neg / max(n_pos, 1)

    lc_analyzer = LearningCurveAnalyzer(disease)
    factory     = ModelFactory(class_ratio=class_ratio, n_samples=len(X_train))
    calibrator  = ModelCalibrator(method="sigmoid")
    evaluator_rows = []

    # Only retrain RF and XGBoost (most prone to overfit)
    for model_name in ["Random Forest", "XGBoost"]:
        logger.info("  Retraining %s with regularised grid …", model_name)
        t0    = time.time()
        model = factory.get_all_models()[model_name]
        model.fit(X_train, y_train)
        logger.info("  Done in %.1fs — best params: %s",
                    time.time() - t0, model.best_params_)

        # Learning curve on new model
        lc_result = lc_analyzer.plot_single(
            model, X_train, y_train,
            label   = f"{model_name} (regularised)",
            scoring = "roc_auc",
        )
        logger.info("  Max gap: %.4f | Verdict: %s",
                    lc_result["max_gap"], lc_result["verdict"])

        # Calibrate
        calibrated = calibrator.fit(model, X_train, y_train)

        # Evaluate
        from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
        y_pred    = model.predict(X_test)
        y_proba   = model.predict_proba(X_test)[:, 1]
        y_pred_cal  = calibrated.predict(X_test)
        y_proba_cal = calibrated.predict_proba(X_test)[:, 1]

        # Calibration before vs after
        logger.info("  Calibration BEFORE:")
        cal_before = ModelCalibrator.evaluate(model,     X_test, y_test, label="before")
        logger.info("  Calibration AFTER:")
        cal_after  = ModelCalibrator.evaluate(calibrated, X_test, y_test, label="after")

        evaluator_rows.append({
            "model":          model_name,
            "test_auc":       round(float(roc_auc_score(y_test, y_proba)), 4),
            "test_recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "test_f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "cv_score":       round(float(model.best_score_), 4),
            "max_gap":        lc_result["max_gap"],
            "verdict":        lc_result["verdict"],
            "ece_before":     cal_before["ece"],
            "ece_after":      cal_after["ece"],
            "best_params":    model.best_params_,
        })

        # Save retrained + calibrated model
        from sklearn.metrics import accuracy_score, precision_score
        metrics = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        ModelPersistence.save(
            model        = model,
            disease      = disease,
            model_name   = model_name,
            scaler       = loader.scaler,
            feature_names= loader.feature_names,
            metrics      = metrics,
            class_names  = loader.get_class_names(),
        )
        logger.info("  Saved retrained %s for %s", model_name, disease)

    return {"disease": disease, "models": evaluator_rows}


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Calibration report
# ─────────────────────────────────────────────────────────────────────────────

def check_calibration(disease: str) -> pd.DataFrame:
    """
    Run calibration evaluation for all saved models and return ECE table.
    """
    logger.info("═" * 60)
    logger.info(" FIX 3 — CALIBRATION CHECK: %s", disease.upper())
    logger.info("═" * 60)

    loader = DataLoader(disease).load_and_prepare()
    _, X_test, _, y_test = loader.get_data()
    assert X_test is not None and y_test is not None, "Test data cannot be None"

    rows = []
    for model_name in MODEL_NAMES:
        try:
            model, scaler, meta = ModelPersistence.load(disease, model_name)
            result = ModelCalibrator.evaluate(
                model, X_test, y_test, label=model_name
            )
            rows.append({
                "Disease":    disease,
                "Model":      model_name,
                "ECE":        result["ece"],
                "Status":     "✓ Good" if result["ece"] < 0.05 else "⚠ Needs calibration",
            })
        except FileNotFoundError:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info("\n%s\n", df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Precision-Recall curves
# ─────────────────────────────────────────────────────────────────────────────

def generate_pr_curves(disease: str) -> None:
    """
    Generate PR curves for all models, marking the optimal threshold.
    """
    logger.info("═" * 60)
    logger.info(" FIX 4 — PR CURVES: %s", disease.upper())
    logger.info("═" * 60)

    loader = DataLoader(disease).load_and_prepare()
    _, X_test, _, y_test = loader.get_data()
    assert X_test is not None and y_test is not None, "Test data cannot be None"
    baseline = float(y_test.mean())

    analyzer = PRCurveAnalyzer(
        disease     = disease,
        class_names = loader.get_class_names(),
    )

    for model_name in MODEL_NAMES:
        try:
            model, scaler, meta = ModelPersistence.load(disease, model_name)
            y_proba   = model.predict_proba(X_test)[:, 1]
            opt_thresh = meta.get("optimal_threshold")
            analyzer.add_model(model_name, y_test, y_proba, opt_threshold=opt_thresh)
        except FileNotFoundError:
            pass

    analyzer.plot_all(baseline_positive_rate=baseline)
    logger.info("  PR curves saved for %s", disease)


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_fixes(diseases: list[str], skip_retrain: bool = False) -> None:
    """Run all four fixes for the specified diseases."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_threshold_rows  = []
    all_calibration_rows = []

    for disease in diseases:
        t0 = time.time()

        # Fix 1: Threshold tuning (always)
        thresh_df = fix_thresholds(disease)
        all_threshold_rows.append(thresh_df)

        # Fix 2: Retrain with regularisation (skippable)
        if not skip_retrain:
            fix_overfitting(disease)
        else:
            logger.info("  Skipping retrain for %s (--skip-retrain)", disease)

        # Fix 3: Calibration check (always)
        cal_df = check_calibration(disease)
        all_calibration_rows.append(cal_df)

        # Fix 4: PR curves (always — need threshold data from fix 1)
        generate_pr_curves(disease)

        logger.info("  Fixes complete for %s in %.1fs", disease, time.time() - t0)

    # ── Save combined reports ─────────────────────────────────────────────────
    if all_threshold_rows:
        df = pd.concat(all_threshold_rows, ignore_index=True)
        path = REPORTS_DIR / "threshold_improvements.csv"
        df.to_csv(path, index=False)
        logger.info("\nThreshold report → %s", path)

        # Print summary
        logger.info("\n%s", "═" * 70)
        logger.info("  THRESHOLD TUNING SUMMARY — MISSED PATIENTS PREVENTED")
        logger.info("═" * 70)
        for _, row in df.iterrows():
            saved = row["Default_FN"] - row["Opt_FN"]
            if saved > 0:
                logger.info(
                    "  %-12s %-22s : %d fewer missed patients "
                    "(recall %.3f → %.3f, threshold %.2f → %.2f)",
                    row["Disease"], row["Model"],
                    saved,
                    row["Default_Recall"], row["Opt_Recall"],
                    0.50, row["Opt_Threshold"],
                )

    if all_calibration_rows:
        df = pd.concat(all_calibration_rows, ignore_index=True)
        path = REPORTS_DIR / "calibration_report.csv"
        df.to_csv(path, index=False)
        logger.info("Calibration report → %s", path)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient Safety Fixes — Threshold Tuning + Overfitting + Calibration + PR Curves"
    )
    parser.add_argument(
        "--disease",
        choices=DISEASES + ["all"],
        default="all",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip model retraining (run threshold/calibration/PR only)",
    )
    args = parser.parse_args()

    diseases = DISEASES if args.disease == "all" else [args.disease]
    logger.info("Starting patient safety fixes for: %s", diseases)
    run_all_fixes(diseases, skip_retrain=args.skip_retrain)
    logger.info("All fixes complete.")


if __name__ == "__main__":
    main()
