"""
evaluation/pr_curves.py
------------------------
Precision-Recall curve analysis — the RIGHT primary metric for
imbalanced medical datasets (diabetes, breast cancer).

Why PR > ROC for imbalanced data
----------------------------------
ROC-AUC uses True Positive Rate vs False Positive Rate.
On imbalanced data (e.g. 65% negative), the FPR denominator is large,
so even a model with many false positives can look great on ROC.

PR-AUC (Average Precision) directly shows the cost of false positives
relative to the number of true positives caught — much more honest.

Rule of thumb: if class imbalance > 1.5:1, always report PR-AUC.

Visualisations produced
-----------------------
1. PR curves for all 4 models (overlaid)
2. F1 iso-lines on the same plot showing optimal operating point
3. Threshold vs Precision/Recall trade-off curve
4. Summary table: PR-AUC + optimal threshold per model
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class PRCurveAnalyzer:
    """
    Generates Precision-Recall curves and operating-point analysis.

    Parameters
    ----------
    disease     : str
    class_names : list[str]
    figures_dir : Path | None
    """

    def __init__(
        self,
        disease:     str,
        class_names: list[str],
        figures_dir: Path | None = None,
    ) -> None:
        self.disease     = disease
        self.class_names = class_names
        self.figures_dir = figures_dir or (FIGURES_DIR / disease)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._results: dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────────────────

    def add_model(
        self,
        model_name: str,
        y_true:     np.ndarray,
        y_proba:    np.ndarray,
        opt_threshold: float | None = None,
    ) -> dict:
        """
        Register a model's probabilities for plotting.

        Parameters
        ----------
        opt_threshold : if provided, mark this threshold on the curve
        """
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_proba)
        ap   = average_precision_score(y_true, y_proba)

        # F1 at each threshold
        f1_arr = np.where(
            prec_arr[:-1] + rec_arr[:-1] > 0,
            2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]),
            0.0,
        )
        best_f1_idx = int(np.argmax(f1_arr))
        best_thresh = float(thresh_arr[best_f1_idx])

        result = {
            "precision":          prec_arr,
            "recall":             rec_arr,
            "thresholds":         thresh_arr,
            "f1":                 f1_arr,
            "ap":                 round(ap, 4),
            "best_f1_threshold":  round(best_thresh, 3),
            "best_f1":            round(float(f1_arr[best_f1_idx]), 4),
            "opt_threshold":      opt_threshold,
        }
        self._results[model_name] = result
        logger.info("  PR-AUC [%s / %-22s] = %.4f  (best-F1 t=%.3f)",
                    self.disease, model_name, ap, best_thresh)
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def plot_all(self, baseline_positive_rate: float) -> None:
        """
        Generate all PR visualisations and save them.

        Parameters
        ----------
        baseline_positive_rate : fraction of positives in the dataset
            (the "no-skill" PR baseline = this value)
        """
        self._plot_pr_curves(baseline_positive_rate)
        self._plot_threshold_tradeoff()
        self._log_summary_table()

    # ─────────────────────────────────────────────────────────────────────────

    def _plot_pr_curves(self, baseline: float) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = plt.cm.tab10(np.linspace(0, 0.8, len(self._results)))

        # F1 iso-lines (background guide)
        f1_values = [0.4, 0.5, 0.6, 0.7, 0.8]
        for f1_val in f1_values:
            rec_line  = np.linspace(0.01, 1.0, 200)
            prec_line = f1_val * rec_line / (2 * rec_line - f1_val)
            mask = (prec_line > 0) & (prec_line <= 1)
            ax.plot(rec_line[mask], prec_line[mask],
                    "--", color="lightgray", lw=0.8, zorder=0)
            ax.text(rec_line[mask][-1] + 0.01,
                    prec_line[mask][-1],
                    f"F1={f1_val}", fontsize=6, color="gray", va="center")

        # No-skill baseline
        ax.axhline(baseline, color="gray", linestyle=":", lw=1.5,
                   label=f"No-skill baseline ({baseline:.2f})")

        for (name, res), color in zip(self._results.items(), colors):
            ax.plot(res["recall"], res["precision"],
                    lw=2, color=color,
                    label=f"{name}  (AP={res['ap']:.3f})")

            # Mark optimal threshold point
            t_opt = res.get("opt_threshold") or res["best_f1_threshold"]
            idx   = np.argmin(np.abs(res["thresholds"] - t_opt))
            ax.scatter(res["recall"][idx], res["precision"][idx],
                       s=80, color=color, zorder=5, marker="D")

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision (PPV)")
        ax.set_title(
            f"{self.disease.replace('_',' ').title()} — Precision-Recall Curves\n"
            f"◆ = optimal threshold  |  dashed = F1 iso-lines",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        self._save(fig, "pr_curves")

    def _plot_threshold_tradeoff(self) -> None:
        """
        For each model: threshold vs Precision, Recall, F1 on the same axes.
        Shows exactly why the default 0.5 threshold is suboptimal.
        """
        n     = len(self._results)
        cols  = min(2, n)
        rows  = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4.5))
        axes  = np.array(axes).flatten()

        for i, (name, res) in enumerate(self._results.items()):
            thresh = res["thresholds"]
            prec   = res["precision"][:-1]
            rec    = res["recall"][:-1]
            f1     = res["f1"]

            axes[i].plot(thresh, prec, color="#1565c0", lw=2, label="Precision")
            axes[i].plot(thresh, rec,  color="#c62828", lw=2, label="Recall")
            axes[i].plot(thresh, f1,   color="#2e7d32", lw=2, label="F1")

            # Default threshold
            axes[i].axvline(0.5, color="black", linestyle="--",
                            lw=1.2, label="Default (0.50)")

            # Optimal threshold
            t_opt = res.get("opt_threshold") or res["best_f1_threshold"]
            axes[i].axvline(t_opt, color="orange", linestyle="-",
                            lw=2, label=f"Optimal ({t_opt:.3f})")

            # Shade the improvement region
            if t_opt < 0.5:
                axes[i].axvspan(t_opt, 0.5, alpha=0.07, color="orange")

            axes[i].set_xlim(0.05, 0.75)
            axes[i].set_ylim(0, 1.05)
            axes[i].set_xlabel("Decision Threshold")
            axes[i].set_ylabel("Score")
            axes[i].set_title(name, fontweight="bold")
            axes[i].legend(fontsize=8)
            axes[i].grid(alpha=0.25)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"{self.disease.replace('_',' ').title()} — Threshold vs Precision / Recall / F1",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, "threshold_tradeoff")

    def _log_summary_table(self) -> None:
        header = f"  {'Model':<24} {'PR-AUC':>8} {'Best-F1 t':>10} {'Best F1':>8}"
        logger.info("\n  PR Curve Summary — %s\n%s\n%s",
                    self.disease, header, "  " + "─"*52)
        for name, res in self._results.items():
            logger.info("  %-24s %8.4f %10.3f %8.4f",
                        name, res["ap"], res["best_f1_threshold"], res["best_f1"])

    def _save(self, fig, name: str) -> None:
        path = self.figures_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("  Saved: %s", path)
