"""
evaluation/shap_explainer.py
-----------------------------
SHAP (SHapley Additive exPlanations) for per-prediction interpretability.

Why SHAP matters in healthcare
-------------------------------
Clinicians and regulators need to know WHY a model flagged a patient
as high-risk. SHAP provides:
  • Global importance  — which features drive the model overall
  • Local explanation  — which features pushed THIS patient's score up/down
  • Direction          — does high glucose increase or decrease risk?

Plots generated
---------------
1. Summary dot plot  — global feature importance + direction
2. Bar summary       — mean |SHAP| per feature
3. Single-patient waterfall — for the highest-risk test patient
4. Dependence plot   — top feature vs SHAP value (interaction shown)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    Wraps SHAP explanation for tree or linear models.

    Parameters
    ----------
    disease      : str
    model_name   : str
    class_names  : list[str]
    figures_dir  : Path | None
    """

    def __init__(
        self,
        disease:      str,
        model_name:   str,
        class_names:  list[str],
        figures_dir:  Path | None = None,
    ) -> None:
        self.disease      = disease
        self.model_name   = model_name
        self.class_names  = class_names
        self.figures_dir  = figures_dir or (FIGURES_DIR / disease)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────

    def explain(
        self,
        model,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        feature_names: list[str],
        max_display: int = 15,
    ) -> None:
        """
        Compute SHAP values and save all explanation plots.

        Parameters
        ----------
        model         : fitted estimator (best_estimator_ is extracted if GridSearchCV)
        X_train       : training data (used to build SHAP background)
        X_test        : test data (used for explanations)
        feature_names : list of feature name strings
        max_display   : max features shown in summary plots
        """
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed — skipping SHAP plots (pip install shap)")
            return

        best = getattr(model, "best_estimator_", model)
        model_type = type(best).__name__

        logger.info("  Computing SHAP values for %s / %s …", self.disease, self.model_name)

        # ── Choose explainer based on model type ──────────────────────────
        try:
            if "RandomForest" in model_type or "XGB" in model_type or "GradientBoosting" in model_type:
                explainer  = shap.TreeExplainer(best)
                shap_vals  = explainer.shap_values(X_test)
                # Tree explainer returns list [class0, class1] for binary classification
                if isinstance(shap_vals, list):
                    sv = shap_vals[1]   # positive class
                else:
                    sv = shap_vals
            else:
                # Linear / SVM: use KernelExplainer with a background summary
                background = shap.kmeans(X_train, min(50, len(X_train)))
                explainer  = shap.KernelExplainer(best.predict_proba, background)
                shap_vals  = explainer.shap_values(X_test[:100], nsamples=100)
                sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        except Exception as exc:
            logger.warning("  SHAP explainer failed for %s: %s", model_type, exc)
            return

        import pandas as pd
        X_df = pd.DataFrame(X_test, columns=feature_names)

        slug = self.model_name.lower().replace(" ", "_")

        # ── Plot 1: Dot summary (global + direction) ──────────────────────
        fig, ax = plt.subplots(figsize=(9, max(5, max_display // 2)))
        shap.summary_plot(
            sv, X_df,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        plt.title(
            f"{self.disease.replace('_',' ').title()} — SHAP Summary ({self.model_name})",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(plt.gcf(), f"shap_summary_{slug}")

        # ── Plot 2: Bar summary (mean |SHAP|) ─────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(9, max(5, max_display // 2)))
        shap.summary_plot(
            sv, X_df,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"{self.disease.replace('_',' ').title()} — Mean |SHAP| ({self.model_name})",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(plt.gcf(), f"shap_bar_{slug}")

        # ── Plot 3: Waterfall for highest-risk patient ────────────────────
        try:
            # Find the test patient with the highest predicted probability
            probas = best.predict_proba(X_test)[:, 1]
            high_risk_idx = int(np.argmax(probas))

            exp = shap.Explanation(
                values        = sv[high_risk_idx],
                base_values   = explainer.expected_value[1]
                                if isinstance(explainer.expected_value, (list, np.ndarray))
                                else explainer.expected_value,
                data          = X_test[high_risk_idx],
                feature_names = feature_names,
            )
            fig3, ax3 = plt.subplots(figsize=(9, max(5, max_display // 2)))
            shap.plots.waterfall(exp, max_display=max_display, show=False)
            plt.title(
                f"Highest-Risk Patient (p={probas[high_risk_idx]:.3f}) — {self.model_name}",
                fontsize=11, fontweight="bold",
            )
            plt.tight_layout()
            self._save(plt.gcf(), f"shap_waterfall_{slug}")
        except Exception as exc:
            logger.debug("  Waterfall plot skipped: %s", exc)

        # ── Plot 4: Dependence plot (top feature) ─────────────────────────
        try:
            top_feat_idx  = int(np.abs(sv).mean(0).argmax())
            top_feat_name = feature_names[top_feat_idx]
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            shap.dependence_plot(
                top_feat_idx, sv, X_df,
                ax=ax4, show=False,
                alpha=0.6,
            )
            ax4.set_title(
                f"{self.disease.replace('_',' ').title()} — SHAP Dependence: {top_feat_name}",
                fontsize=11, fontweight="bold",
            )
            plt.tight_layout()
            self._save(fig4, f"shap_dependence_{slug}")
        except Exception as exc:
            logger.debug("  Dependence plot skipped: %s", exc)

        logger.info("  SHAP plots saved for %s / %s", self.disease, self.model_name)

    # ─────────────────────────────────────────────────────────────────────────

    def _save(self, fig, name: str) -> None:
        path = self.figures_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
        logger.debug("  Saved: %s", path)
