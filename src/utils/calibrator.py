"""
utils/calibrator.py
--------------------
Calibrates model probabilities so that predicted P(y=1) = 0.8
actually means ~80% of similarly-scored patients are positive.

Why calibration matters in healthcare
--------------------------------------
Raw classifier outputs (especially tree ensembles) are often poorly
calibrated — they may systematically over- or under-state risk:

  • Overconfident model: says "90% chance of diabetes" but only 60% of
    those patients actually have it → clinician overtreats.
  • Underconfident: says "30% chance" when 50% positive → screens pass
    patients who need follow-up.

We use sklearn's CalibratedClassifierCV with:
  • method='sigmoid'  — Platt scaling (good for SVM, LR)
  • method='isotonic' — non-parametric (better for RF, XGBoost with
                        enough data, but prone to overfit on small sets)

The calibrator is fit on a held-out CALIBRATION split (20% of train)
to avoid leakage — NOT on the test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelCalibrator:
    """
    Wraps a trained classifier in CalibratedClassifierCV.

    Parameters
    ----------
    method : 'sigmoid' | 'isotonic'
        Platt scaling (fewer samples) or isotonic regression (more data).
    cal_size : float
        Fraction of training data to hold out for calibration fitting.
    """

    def __init__(
        self,
        method:   str   = "sigmoid",
        cal_size: float = 0.20,
    ) -> None:
        self.method   = method
        self.cal_size = cal_size

    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        base_model,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> CalibratedClassifierCV:
        """
        Fit a calibration wrapper around a pre-trained model.

        Uses a separate calibration split from training data.
        Returns a fully-fitted calibrated classifier.
        """
        best_est = (
            base_model.best_estimator_
            if hasattr(base_model, "best_estimator_")
            else base_model
        )

        # Hold out a calibration fold from training data (no leakage)
        X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(
            X_train, y_train,
            test_size   = self.cal_size,
            stratify    = y_train,
            random_state= RANDOM_STATE,
        )

        # Refit the estimator on the reduced train fold, then calibrate on val
        calibrated = CalibratedClassifierCV(
            estimator = best_est,
            method    = self.method,
            cv        = "prefit",       # base model already fitted
        )

        # Refit base on smaller fold so calibrator sees fresh predictions
        best_est.fit(X_cal_train, y_cal_train)
        calibrated.fit(X_cal_val, y_cal_val)

        logger.debug(
            "  Calibrated with method='%s' on %d samples",
            self.method, len(X_cal_val),
        )
        return calibrated

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def evaluate(
        model,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        n_bins:  int = 5,
        label:   str = "",
    ) -> dict:
        """
        Compute calibration error and return bin-level stats.

        Returns dict with:
          mean_predicted_values, fraction_of_positives, ece (expected calibration error)
        """
        proba = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins)

        # Expected Calibration Error — weighted average absolute difference
        bin_counts = np.histogram(proba, bins=n_bins)[0]
        weights    = bin_counts / bin_counts.sum()
        ece        = float(np.average(np.abs(frac_pos - mean_pred), weights=weights[:len(frac_pos)]))

        lines = [f"  {'Pred prob':>10}  {'Actual rate':>11}  {'Diff':>7}"]
        lines.append(f"  {'─'*10}  {'─'*11}  {'─'*7}")
        for mp, fp in zip(mean_pred, frac_pos):
            diff = fp - mp
            flag = "↑ under" if diff > 0.05 else ("↓ over" if diff < -0.05 else "✓ ok")
            lines.append(f"  {mp:>10.3f}  {fp:>11.3f}  {diff:>+7.3f}  {flag}")
        lines.append(f"  ECE = {ece:.4f}  {'(lower = better)':}")

        logger.info("  Calibration check%s:\n%s", f" [{label}]" if label else "", "\n".join(lines))
        return {
            "mean_predicted_values": mean_pred.tolist(),
            "fraction_of_positives": frac_pos.tolist(),
            "ece": round(ece, 4),
        }
