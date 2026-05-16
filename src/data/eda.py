"""
data/eda.py
-----------
Exploratory Data Analysis for the Disease Prediction System.

Generates and saves:
  • class distribution bar chart
  • correlation heatmap
  • feature distributions by class
  • box plots for numeric features
  • pairplot of top correlated features
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Aesthetics ────────────────────────────────────────────────────────────────
PALETTE   = ["#2196F3", "#F44336"]   # blue = negative, red = positive
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)


class EDAAnalyzer:
    """
    Runs a suite of EDA plots for a given disease dataset.

    Parameters
    ----------
    df           : pd.DataFrame — full cleaned dataframe (before encoding).
    disease      : str          — used for titles and file naming.
    target_col   : str          — name of the target column (already numeric 0/1).
    class_names  : list[str]    — human-readable class labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        disease: str,
        target_col: str,
        class_names: list[str],
    ) -> None:
        self.df          = df.copy()
        self.disease     = disease
        self.target_col  = target_col
        self.class_names = class_names
        self.figures_dir = FIGURES_DIR / disease
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.feature_cols = [c for c in df.columns if c != target_col]
        self.numeric_cols = df[self.feature_cols].select_dtypes(
            include=[np.number]
        ).columns.tolist()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Generate and save all EDA plots."""
        logger.info("[EDA] Running analysis for %s …", self.disease.upper())
        self.plot_class_distribution()
        self.plot_correlation_heatmap()
        self.plot_feature_distributions()
        self.plot_box_plots()
        self.plot_top_correlated_features()
        logger.info("[EDA] All plots saved to %s", self.figures_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Individual Plots
    # ─────────────────────────────────────────────────────────────────────────

    def plot_class_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = self.df[self.target_col].value_counts().sort_index()
        bars = ax.bar(self.class_names, counts.values, color=PALETTE, width=0.5, edgecolor="white")
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{count}\n({100 * count / len(self.df):.1f}%)",
                ha="center", va="bottom", fontsize=11
            )
        ax.set_title(f"{self.disease.replace('_', ' ').title()} — Class Distribution", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Patients")
        ax.set_xlabel("Diagnosis")
        self._save(fig, "class_distribution")

    def plot_correlation_heatmap(self) -> None:
        numeric_df = self.df[self.numeric_cols + [self.target_col]]
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(max(8, len(corr) // 2), max(6, len(corr) // 2)))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5,
            annot_kws={"size": 7}, ax=ax
        )
        ax.set_title(
            f"{self.disease.replace('_', ' ').title()} — Feature Correlation Heatmap",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        self._save(fig, "correlation_heatmap")

    def plot_feature_distributions(self) -> None:
        """Histograms of each numeric feature, coloured by class."""
        cols = min(4, len(self.numeric_cols))
        rows = (len(self.numeric_cols) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.array(axes).flatten()

        for i, col in enumerate(self.numeric_cols):
            for label, cname in enumerate(self.class_names):
                subset = self.df[self.df[self.target_col] == label][col]
                axes[i].hist(subset, bins=25, alpha=0.6, label=cname, color=PALETTE[label])
            axes[i].set_title(col, fontsize=9, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].legend(fontsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"{self.disease.replace('_', ' ').title()} — Feature Distributions by Class",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        self._save(fig, "feature_distributions")

    def plot_box_plots(self) -> None:
        """Box plots showing feature spread and outliers by class."""
        cols = min(4, len(self.numeric_cols))
        rows = (len(self.numeric_cols) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.array(axes).flatten()

        for i, col in enumerate(self.numeric_cols):
            data = [
                self.df[self.df[self.target_col] == label][col].values
                for label in [0, 1]
            ]
            bp = axes[i].boxplot(data, patch_artist=True, widths=0.5)
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[i].set_xticklabels(self.class_names, fontsize=8)
            axes[i].set_title(col, fontsize=9, fontweight="bold")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"{self.disease.replace('_', ' ').title()} — Box Plots by Class",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        self._save(fig, "box_plots")

    def plot_top_correlated_features(self, top_n: int = 6) -> None:
        """
        Scatter matrix of the top-N features most correlated with the target.
        """
        corr_with_target = (
            self.df[self.numeric_cols + [self.target_col]]
            .corr()[self.target_col]
            .drop(self.target_col)
            .abs()
            .sort_values(ascending=False)
        )
        top_features = corr_with_target.head(top_n).index.tolist()

        plot_df = self.df[top_features + [self.target_col]].copy()
        plot_df["Class"] = plot_df[self.target_col].map(
            {i: name for i, name in enumerate(self.class_names)}
        )

        pp = sns.pairplot(
            plot_df.drop(columns=[self.target_col]),
            hue="Class",
            palette=dict(zip(self.class_names, PALETTE)),
            plot_kws={"alpha": 0.5, "s": 20},
            diag_kind="kde",
        )
        pp.fig.suptitle(
            f"{self.disease.replace('_', ' ').title()} — Top {top_n} Features Pairplot",
            y=1.02, fontsize=13, fontweight="bold"
        )
        self._save(pp.fig, "top_features_pairplot")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.figures_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("  Saved: %s", path)
