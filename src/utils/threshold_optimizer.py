"""
utils/threshold_optimizer.py
------------------------------
Finds the medically-optimal decision threshold for each classifier.

Core problem
------------
sklearn's default threshold is 0.5 (predict positive when P(y=1) ≥ 0.5).
This maximises accuracy on balanced data, but in medical screening:

  • A False Negative (missed disease) → patient goes untreated → worse outcome
  • A False Positive  (healthy flagged) → extra tests, anxiety, cost

For screening tools we deliberately lower the threshold so the model is
more willing to flag a patient as at-risk.  The optimal threshold is the
one that achieves a target recall (e.g. ≥ 0.85) while maximising F1.

Strategy
--------
1. Sweep thresholds 0.10 → 0.75 in 0.025 steps.
2. Keep only rows where recall ≥ RECALL_TARGET.
3. Among those, pick the threshold with the highest F1.
4. Fall back to maximum recall if the target is unreachable.

Outputs
-------
• Returns a ThresholdResult with all stats.
• Saves the optimal threshold into the model's .meta.json so the
  prediction API can load it automatically without retraining.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Per-disease recall targets ────────────────────────────────────────────────
# These encode clinical priority: breast cancer and diabetes screening must
# catch most positive cases; heart disease dataset is already near-perfect.
RECALL_TARGETS: dict[str, float] = {
    "heart":        0.92,   # already excellent; keep FN minimal
    "diabetes":     0.85,   # priority: catch diabetics early
    "breast_cancer": 0.95,  # priority: never miss malignancy
}

DEFAULT_RECALL_TARGET = 0.85


@dataclass
class ThresholdResult:
    disease:         str
    model_name:      str
    optimal_threshold: float
    recall:          float
    precision:       float
    f1:              float
    false_negatives: int
    false_positives: int
    true_positives:  int
    true_negatives:  int
    total_positives: int
    default_recall:  float          # recall at threshold=0.50 for comparison
    default_fn:      int            # FN at threshold=0.50 for comparison
    recall_gain:     float          # recall improvement vs default


class ThresholdOptimizer:
    """
    Finds the optimal prediction threshold for a trained classifier.

    Parameters
    ----------
    disease      : str  — "heart" | "diabetes" | "breast_cancer"
    model_name   : str  — e.g. "XGBoost"
    recall_target: float | None — override per-disease default
    """

    def __init__(
        self,
        disease:      str,
        model_name:   str,
        recall_target: float | None = None,
    ) -> None:
        self.disease      = disease
        self.model_name   = model_name
        self.recall_target = (
            recall_target
            if recall_target is not None
            else RECALL_TARGETS.get(disease, DEFAULT_RECALL_TARGET)
        )

    # ─────────────────────────────────────────────────────────────────────────

    def optimize(
        self,
        y_true:  np.ndarray,
        y_proba: np.ndarray,
    ) -> ThresholdResult:
        """
        Run threshold sweep and return the optimal ThresholdResult.

        Parameters
        ----------
        y_true  : ground-truth labels (0/1)
        y_proba : predicted probabilities for the positive class
        """
        thresholds = np.arange(0.10, 0.76, 0.025)

        # ── Default stats (t = 0.50) ──────────────────────────────────────
        default_pred  = (y_proba >= 0.50).astype(int)
        default_rec   = recall_score(y_true, default_pred, zero_division=0)
        default_fn    = int(((default_pred == 0) & (y_true == 1)).sum())

        # ── Sweep ─────────────────────────────────────────────────────────
        rows = []
        for t in thresholds:
            pred = (y_proba >= t).astype(int)
            rec  = recall_score(y_true, pred, zero_division=0)
            prec = precision_score(y_true, pred, zero_division=0)
            f1   = f1_score(y_true, pred, zero_division=0)
            fn   = int(((pred == 0) & (y_true == 1)).sum())
            fp   = int(((pred == 1) & (y_true == 0)).sum())
            tp   = int(((pred == 1) & (y_true == 1)).sum())
            tn   = int(((pred == 0) & (y_true == 0)).sum())
            rows.append((round(float(t), 3), rec, prec, f1, fn, fp, tp, tn))

        # ── Select optimal ────────────────────────────────────────────────
        # Priority: meet recall target → highest F1 among those rows
        # Fallback: maximum recall
        eligible = [r for r in rows if r[1] >= self.recall_target]
        if eligible:
            best = max(eligible, key=lambda r: r[3])
            logger.info(
                "  [%s/%s] Optimal threshold: %.3f "
                "(recall=%.3f ≥ target=%.2f, F1=%.3f)",
                self.disease, self.model_name,
                best[0], best[1], self.recall_target, best[3],
            )
        else:
            best = max(rows, key=lambda r: r[1])
            logger.warning(
                "  [%s/%s] Could not meet recall target %.2f; "
                "best achievable recall=%.3f at t=%.3f",
                self.disease, self.model_name,
                self.recall_target, best[1], best[0],
            )

        t_opt, rec, prec, f1, fn, fp, tp, tn = best
        total_pos = int((y_true == 1).sum())

        result = ThresholdResult(
            disease           = self.disease,
            model_name        = self.model_name,
            optimal_threshold = t_opt,
            recall            = round(rec,  4),
            precision         = round(prec, 4),
            f1                = round(f1,   4),
            false_negatives   = fn,
            false_positives   = fp,
            true_positives    = tp,
            true_negatives    = tn,
            total_positives   = total_pos,
            default_recall    = round(default_rec, 4),
            default_fn        = default_fn,
            recall_gain       = round(rec - default_rec, 4),
        )

        self._persist(result)
        self._log_summary(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _persist(self, result: ThresholdResult) -> None:
        """Write optimal threshold into the model's metadata JSON."""
        slug      = self.model_name.lower().replace(" ", "_")
        meta_path = MODELS_DIR / self.disease / f"{slug}.meta.json"

        if not meta_path.exists():
            logger.warning("  Meta file not found: %s — skipping persist", meta_path)
            return

        meta = json.loads(meta_path.read_text())
        meta["optimal_threshold"]  = result.optimal_threshold
        meta["recall_target"]      = self.recall_target
        meta["threshold_metrics"]  = {
            "recall":          result.recall,
            "precision":       result.precision,
            "f1":              result.f1,
            "false_negatives": result.false_negatives,
            "false_positives": result.false_positives,
            "total_positives": result.total_positives,
            "default_recall":  result.default_recall,
            "default_fn":      result.default_fn,
            "recall_gain":     result.recall_gain,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.debug("  Threshold persisted → %s", meta_path)

    def _log_summary(self, r: ThresholdResult) -> None:
        logger.info(
            "  ┌─ Threshold change for %s / %s\n"
            "  │  Default  (t=0.50): recall=%.3f  FN=%d/%d  (%.0f%% missed)\n"
            "  │  Optimal  (t=%.3f): recall=%.3f  FN=%d/%d  (%.0f%% missed)\n"
            "  └─ Improvement: +%.3f recall  |  -%d fewer missed patients",
            r.disease, r.model_name,
            r.default_recall, r.default_fn, r.total_positives,
            100 * r.default_fn / max(r.total_positives, 1),
            r.optimal_threshold, r.recall, r.false_negatives, r.total_positives,
            100 * r.false_negatives / max(r.total_positives, 1),
            r.recall_gain, r.default_fn - r.false_negatives,
        )
