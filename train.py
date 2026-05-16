"""
train.py
---------
Master training script — full ML pipeline for all three disease datasets.

Pipeline per disease
--------------------
1.  Data loading, cleaning, imputation, outlier capping
2.  Feature engineering  (diabetes: BMI_category, Glucose_Insulin)
3.  Stratified train/test split (80/20)
4.  StandardScaler — fit on train only (no leakage)
5.  SMOTE — applied to training fold only when imbalance > 1.3:1
6.  Train 4 base classifiers (GridSearchCV × 5-fold CV)
7.  Train stacking ensemble (RF + XGB + SVM → LR meta-learner)
8.  Evaluate all 5 models — Accuracy, Precision, Recall, F1, ROC-AUC
9.  Calibration curves, ROC curves, Confusion matrices, PR curves,
    Feature importance, Threshold trade-off plots
10. SHAP explainability (summary, bar, waterfall, dependence)
11. Threshold optimisation — persist optimal threshold to meta.json
12. Save all models (.joblib + .meta.json)
13. Write CSV reports

Usage
-----
    python train.py                          # all diseases, all features
    python train.py --disease diabetes       # one disease
    python train.py --no-shap                # skip SHAP (faster)
    python train.py --no-smote               # disable SMOTE
    python train.py --no-save                # dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import DATASET_CONFIG, FIGURES_DIR, REPORTS_DIR
from data.data_loader import DataLoader
from data.eda import EDAAnalyzer
from evaluation.evaluator import ModelEvaluator
from evaluation.learning_curves import LearningCurveAnalyzer
from evaluation.pr_curves import PRCurveAnalyzer
from evaluation.shap_explainer import SHAPExplainer
from models.model_factory import ModelFactory
from models.stacking_ensemble import build_stacking_ensemble
from utils.logger import get_logger
from utils.model_persistence import ModelPersistence
from utils.threshold_optimizer import ThresholdOptimizer

logger = get_logger("train")
DISEASES = list(DATASET_CONFIG.keys())


# ─────────────────────────────────────────────────────────────────────────────

def train_disease(
    disease:    str,
    run_eda:    bool = True,
    run_shap:   bool = True,
    use_smote:  bool = True,
    save_models:bool = True,
) -> pd.DataFrame:

    bar = "=" * 64
    logger.info("%s", bar)
    logger.info("  TRAINING: %s", disease.upper().replace("_", " "))
    logger.info("%s", bar)

    # ── 1. Data ───────────────────────────────────────────────────────────
    loader = DataLoader(disease, use_smote=use_smote).load_and_prepare()
    X_train, X_test, y_train, y_test = loader.get_data()
    n_neg = (y_train == 0).sum();  n_pos = (y_train == 1).sum()
    class_ratio = n_neg / max(n_pos, 1)

    # ── 2. EDA ────────────────────────────────────────────────────────────
    if run_eda and loader.clean_df is not None:
        cfg = DATASET_CONFIG[disease]
        EDAAnalyzer(
            df          = loader.clean_df,
            disease     = disease,
            target_col  = cfg["target_column"],
            class_names = cfg["class_names"],
        ).run_all()

    # ── 3. Build models ───────────────────────────────────────────────────
    factory = ModelFactory(class_ratio=class_ratio, n_samples=len(X_train))
    models  = factory.get_all_models()
    models["Stacking Ensemble"] = build_stacking_ensemble(class_ratio)

    evaluator = ModelEvaluator(disease, loader.get_class_names())
    trained:  dict = {}

    # ── 4. Train & evaluate ───────────────────────────────────────────────
    for name, model in models.items():
        logger.info("  Training: %-22s …", name)
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.error("  [ERROR] %s: %s", name, exc)
            continue
        elapsed = time.time() - t0
        cv_score = getattr(model, "best_score_", None)
        logger.info("    %.1fs | CV=%.4f | params=%s",
                    elapsed,
                    cv_score if cv_score else float("nan"),
                    getattr(model, "best_params_", "n/a"))
        evaluator.evaluate(name, model, X_test, y_test)
        trained[name] = model

    # ── 5. Plots ──────────────────────────────────────────────────────────
    evaluator.plot_roc_curves(X_test, y_test)
    evaluator.plot_confusion_matrices(X_test, y_test)
    evaluator.plot_metrics_comparison()
    evaluator.plot_calibration_curves(X_test, y_test)

    for mname in ("Random Forest", "XGBoost", "Stacking Ensemble"):
        if mname in trained:
            evaluator.plot_feature_importance(trained[mname], loader.feature_names)

    # ── 6. PR curves ──────────────────────────────────────────────────────
    pr_analyzer = PRCurveAnalyzer(disease, loader.get_class_names())
    for name, m in evaluator.results.items():
        if m["y_proba"] is not None:
            pr_analyzer.add_model(name, y_test, m["y_proba"])
    pr_analyzer.plot_all(baseline_positive_rate=float(y_test.mean()))

    # ── 7. Learning curves (heart disease overfitting check) ──────────────
    if disease == "heart" and "Random Forest" in trained:
        LearningCurveAnalyzer(disease).plot_single(
            trained["Random Forest"], X_train, y_train,
            label="Random Forest (regularised)",
        )

    # ── 8. SHAP ───────────────────────────────────────────────────────────
    if run_shap:
        for mname in ("Random Forest", "XGBoost"):
            if mname in trained:
                SHAPExplainer(disease, mname, loader.get_class_names()).explain(
                    trained[mname], X_train, X_test, loader.feature_names
                )

    # ── 9. Summary ────────────────────────────────────────────────────────
    summary_df  = evaluator.get_summary_df()
    best_model  = evaluator.get_best_model()
    logger.info("\n%s", summary_df.to_string(index=False))
    logger.info("  ⭐  Best model: %s (ROC-AUC %.4f)",
                best_model, evaluator.results[best_model]["roc_auc"])

    # ── 10. Save + threshold optimisation ────────────────────────────────
    if save_models:
        for name, model in trained.items():
            ModelPersistence.save(
                model         = model,
                disease       = disease,
                model_name    = name,
                scaler        = loader.scaler,
                feature_names = loader.feature_names,
                metrics       = evaluator.results[name],
                class_names   = loader.get_class_names(),
            )
            # Persist optimal threshold into meta.json
            if evaluator.results[name]["y_proba"] is not None:
                ThresholdOptimizer(disease, name).optimize(
                    y_test,
                    evaluator.results[name]["y_proba"],
                )

    # ── 11. Report ────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{disease}_metrics.csv"
    summary_df.to_csv(report_path, index=False)
    logger.info("  Report → %s", report_path)

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Disease Prediction — Training Pipeline")
    parser.add_argument("--disease",  choices=DISEASES + ["all"], default="all")
    parser.add_argument("--no-eda",   action="store_true")
    parser.add_argument("--no-shap",  action="store_true")
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--no-save",  action="store_true")
    args = parser.parse_args()

    diseases = DISEASES if args.disease == "all" else [args.disease]
    t0_all   = time.time()
    summaries = []

    for disease in diseases:
        try:
            df = train_disease(
                disease,
                run_eda     = not args.no_eda,
                run_shap    = not args.no_shap,
                use_smote   = not args.no_smote,
                save_models = not args.no_save,
            )
            df.insert(0, "Disease", disease)
            summaries.append(df)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", disease, exc, exc_info=True)

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(REPORTS_DIR / "all_diseases_metrics.csv", index=False)
        logger.info("\n%s", "=" * 64)
        logger.info("  DONE — %.1fs total", time.time() - t0_all)
        logger.info("\n%s", combined.to_string(index=False))


if __name__ == "__main__":
    main()
