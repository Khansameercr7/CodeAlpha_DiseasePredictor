"""
evaluation/evaluator.py
------------------------
Comprehensive model evaluation with healthcare-focused metrics.

Why each metric matters in healthcare
--------------------------------------
Accuracy    — Baseline correctness; misleading on imbalanced data (always report with others).
Precision   — Of all patients flagged positive, how many truly have the disease?
              Low precision → unnecessary treatments, anxiety, cost.
Recall      — Of all patients WITH the disease, how many did we catch?
              Low recall → missed diagnoses → worst healthcare outcome.
F1-Score    — Harmonic mean of Precision & Recall; balanced view.
ROC-AUC     — Model's ability to distinguish classes at ALL thresholds;
              threshold-independent; ideal for comparing models.
Confusion Matrix — Full breakdown: TP/TN/FP/FN; reveals specific failure modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

PALETTE = ["#2196F3", "#F44336"]
sns.set_theme(style="whitegrid", font_scale=1.05)


class ModelEvaluator:
    """
    Evaluates and compares trained classifiers, saves metric reports and plots.

    Parameters
    ----------
    disease      : str
    class_names  : list[str]
    figures_dir  : Path | None — override default save location
    """

    def __init__(
        self,
        disease: str,
        class_names: list[str],
        figures_dir: Path | None = None,
    ) -> None:
        self.disease     = disease
        self.class_names = class_names
        self.figures_dir = figures_dir or (FIGURES_DIR / disease)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model_name: str,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Compute and store all metrics for one model."""
        y_pred  = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba if y_proba is not None else y_pred), 4),
            "conf_matrix": confusion_matrix(y_test, y_pred),
            "y_proba":   y_proba,
            "y_pred":    y_pred,
            "report":    classification_report(
                y_test, y_pred,
                target_names=self.class_names,
                zero_division=0,
            ),
        }

        if hasattr(model, "best_params_"):
            metrics["best_params"] = model.best_params_
            metrics["cv_score"]    = round(model.best_score_, 4)

        self.results[model_name] = metrics
        logger.info(
            "  %-22s → Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
            model_name,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["roc_auc"],
        )
        return metrics

    def get_best_model(self) -> str:
        """Return the model name with the highest ROC-AUC."""
        return max(self.results, key=lambda m: self.results[m]["roc_auc"])

    def get_summary_df(self) -> pd.DataFrame:
        """Return a DataFrame comparing all evaluated models."""
        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model":     name,
                "Accuracy":  m["accuracy"],
                "Precision": m["precision"],
                "Recall":    m["recall"],
                "F1-Score":  m["f1"],
                "ROC-AUC":   m["roc_auc"],
                "CV-Score":  m.get("cv_score", "-"),
            })
        return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────────

    def plot_confusion_matrices(self, X_test, y_test) -> None:
        """Grid of confusion matrices for all evaluated models."""
        n = len(self.results)
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = np.array(axes).flatten()

        for i, (name, m) in enumerate(self.results.items()):
            disp = ConfusionMatrixDisplay(
                confusion_matrix=m["conf_matrix"],
                display_labels=self.class_names,
            )
            disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
            axes[i].set_title(name, fontsize=11, fontweight="bold")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"{self.disease.replace('_', ' ').title()} — Confusion Matrices",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        self._save(fig, "confusion_matrices")

    def plot_roc_curves(self, X_test, y_test) -> None:
        """Overlay ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(self.results)))

        for (name, m), color in zip(self.results.items(), colors):
            if m["y_proba"] is not None:
                fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
                ax.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})", color=color, lw=2)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(
            f"{self.disease.replace('_', ' ').title()} — ROC Curves",
            fontsize=13, fontweight="bold"
        )
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        self._save(fig, "roc_curves")

    def plot_metrics_comparison(self) -> None:
        """Grouped bar chart comparing Accuracy/Precision/Recall/F1/AUC."""
        df = self.get_summary_df()
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x = np.arange(len(df))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = plt.cm.Set2(np.linspace(0, 0.9, len(metrics)))

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax.bar(x + i * width, df[metric], width, label=metric, color=color)

        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(df["Model"], rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(
            f"{self.disease.replace('_', ' ').title()} — Model Comparison",
            fontsize=13, fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(y=0.8, color="gray", linestyle="--", lw=0.8, alpha=0.5)
        plt.tight_layout()
        self._save(fig, "model_comparison")

    def plot_feature_importance(self, model, feature_names: list[str]) -> None:
        """
        Feature importance from Random Forest or XGBoost.
        Falls back to logistic regression coefficients.
        """
        best = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        name = type(best).__name__

        if hasattr(best, "feature_importances_"):
            importances = best.feature_importances_
        elif hasattr(best, "coef_"):
            importances = np.abs(best.coef_[0])
        else:
            logger.debug("  Feature importance not available for %s", name)
            return

        indices = np.argsort(importances)[::-1][:20]   # top 20
        sorted_names = [feature_names[i] for i in indices]
        sorted_imp   = importances[indices]

        fig, ax = plt.subplots(figsize=(8, max(4, len(indices) // 2)))
        colors = plt.cm.RdYlGn(sorted_imp / sorted_imp.max())
        ax.barh(sorted_names[::-1], sorted_imp[::-1], color=colors[::-1])
        ax.set_xlabel("Importance Score")
        ax.set_title(
            f"{self.disease.replace('_', ' ').title()} ({name}) — Feature Importance",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        self._save(fig, f"feature_importance_{name.lower()}")

    # ─────────────────────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.figures_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("  Saved: %s", path)

    def plot_calibration_curves(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Reliability diagram: predicted probability vs actual positive rate.
        A perfectly calibrated model follows the diagonal.
        """
        from sklearn.calibration import calibration_curve

        fig, ax = plt.subplots(figsize=(7, 6))
        colors  = plt.cm.tab10(np.linspace(0, 0.8, len(self.results)))

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
        ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

        for (name, m), color in zip(self.results.items(), colors):
            if m["y_proba"] is not None:
                frac_pos, mean_pred = calibration_curve(
                    y_test, m["y_proba"], n_bins=6
                )
                ax.plot(mean_pred, frac_pos, "s-", color=color,
                        lw=2, label=name, markersize=6)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (Actual)")
        ax.set_title(
            f"{self.disease.replace('_',' ').title()} — Calibration Curves",
            fontsize=13, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, "calibration_curves")
