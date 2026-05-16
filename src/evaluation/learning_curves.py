"""
evaluation/learning_curves.py
------------------------------
Generates learning curves and train/validation gap plots to diagnose
overfitting and underfitting in trained classifiers.

Why this matters
----------------
A model with train_score=1.0 and val_score=0.99 on the full dataset
looks great — but if at n=200 samples train=1.0 and val=0.94, the gap
is 0.06, indicating the model memorises rather than generalises.

Learning curves reveal:
  • Overfitting  — large gap between train and val that doesn't close
  • Underfitting — both curves plateau at a low score
  • Convergence  — both curves converge: more data won't help
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


class LearningCurveAnalyzer:
    """
    Plots learning curves for one or more estimators on the same axes.

    Parameters
    ----------
    disease     : str
    figures_dir : Path | None
    """

    def __init__(self, disease: str, figures_dir: Path | None = None) -> None:
        self.disease     = disease
        self.figures_dir = figures_dir or (FIGURES_DIR / disease)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────

    def plot_single(
        self,
        estimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        label:   str = "Model",
        scoring: str = "roc_auc",
        n_points: int = 8,
    ) -> dict:
        """
        Generate and save a learning curve for one estimator.

        Returns dict with train/val scores and gap stats.
        """
        best = getattr(estimator, "best_estimator_", estimator)
        train_sizes = np.linspace(0.15, 1.0, n_points)

        sizes, train_scores, val_scores = learning_curve(
            best, X_train, y_train,
            train_sizes  = train_sizes,
            cv           = CV,
            scoring      = scoring,
            n_jobs       = -1,
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)
        gap        = train_mean - val_mean

        # ── Plot ─────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Left: train vs validation curves
        ax1.plot(sizes, train_mean, "o-", color="#1565c0", lw=2, label="Train score")
        ax1.fill_between(sizes,
                         train_mean - train_std,
                         train_mean + train_std,
                         alpha=0.15, color="#1565c0")
        ax1.plot(sizes, val_mean,   "s-", color="#c62828", lw=2, label="Val score (CV)")
        ax1.fill_between(sizes,
                         val_mean - val_std,
                         val_mean + val_std,
                         alpha=0.15, color="#c62828")
        ax1.set_xlabel("Training samples")
        ax1.set_ylabel(scoring.upper().replace("_", "-"))
        ax1.set_title(f"{label} — Learning Curve", fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.set_ylim(max(0, val_mean.min() - 0.05), 1.02)
        ax1.grid(alpha=0.3)

        # Right: generalisation gap
        gap_color = "#e65100" if gap.max() > 0.05 else "#2e7d32"
        ax2.fill_between(sizes, 0, gap, alpha=0.4, color=gap_color)
        ax2.plot(sizes, gap, "o-", color=gap_color, lw=2, label="Train − Val gap")
        ax2.axhline(0.05, color="red",  linestyle="--", lw=1, label="Overfit threshold (0.05)")
        ax2.axhline(0.00, color="gray", linestyle="-",  lw=0.8)
        ax2.set_xlabel("Training samples")
        ax2.set_ylabel("Generalisation Gap")
        ax2.set_title(f"{label} — Overfitting Diagnosis", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.set_ylim(-0.01, max(0.12, gap.max() + 0.02))
        ax2.grid(alpha=0.3)

        verdict = "⚠ OVERFITTING" if gap.max() > 0.05 else "✓ OK"
        fig.suptitle(
            f"{self.disease.replace('_',' ').title()} | {label} | {verdict}  "
            f"(max gap={gap.max():.4f})",
            fontsize=12, fontweight="bold", color=gap_color,
        )
        plt.tight_layout()

        slug = label.lower().replace(" ", "_")
        path = self.figures_dir / f"learning_curve_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Learning curve saved → %s", path)

        return {
            "train_sizes":  sizes.tolist(),
            "train_mean":   train_mean.tolist(),
            "val_mean":     val_mean.tolist(),
            "gap":          gap.tolist(),
            "max_gap":      round(float(gap.max()), 4),
            "final_val":    round(float(val_mean[-1]), 4),
            "verdict":      verdict,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def compare(
        self,
        estimators:  dict[str, object],   # label → fitted estimator
        X_train:     np.ndarray,
        y_train:     np.ndarray,
        scoring:     str = "roc_auc",
    ) -> None:
        """
        Overlay learning curves for multiple estimators (e.g., overfit vs fixed).
        Useful for demonstrating the impact of a regularisation change.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = plt.cm.tab10(np.linspace(0, 0.7, len(estimators)))
        train_sizes_common = np.linspace(0.15, 1.0, 8)

        for (label, est), color in zip(estimators.items(), colors):
            best = getattr(est, "best_estimator_", est)
            sizes, tr_sc, va_sc = learning_curve(
                best, X_train, y_train,
                train_sizes = train_sizes_common,
                cv          = CV,
                scoring     = scoring,
                n_jobs      = -1,
            )
            ax.plot(sizes, va_sc.mean(1), "o-", color=color, lw=2,
                    label=f"{label}  (val={va_sc.mean(1)[-1]:.3f})")
            ax.fill_between(sizes,
                            va_sc.mean(1) - va_sc.std(1),
                            va_sc.mean(1) + va_sc.std(1),
                            alpha=0.12, color=color)

        ax.set_xlabel("Training samples")
        ax.set_ylabel(f"Val {scoring.upper().replace('_','-')}")
        ax.set_title(
            f"{self.disease.replace('_',' ').title()} — Validation Curves Comparison",
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = self.figures_dir / "learning_curve_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Comparison curve saved → %s", path)
