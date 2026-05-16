"""
utils/validators.py
--------------------
Input validation for the prediction API.
Validates patient data before it reaches the model — preventing
garbage-in / garbage-out and surfacing actionable error messages
to API consumers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(ValueError):
    """Raised when patient input fails validation."""
    pass


# ── Per-feature clinical bounds (min, max) ─────────────────────────────────────
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    # Heart
    "age":       (1,   120),
    "sex":       (0,   1),
    "cp":        (0,   3),
    "trestbps":  (60,  250),   # resting blood pressure mmHg
    "chol":      (100, 600),   # serum cholesterol mg/dl
    "fbs":       (0,   1),
    "restecg":   (0,   2),
    "thalach":   (60,  220),   # max heart rate
    "exang":     (0,   1),
    "oldpeak":   (0,   10),
    "slope":     (0,   2),
    "ca":        (0,   4),
    "thal":      (0,   3),

    # Diabetes
    "Pregnancies":               (0,   20),
    "Glucose":                   (40,  400),
    "BloodPressure":             (30,  200),
    "SkinThickness":             (0,   100),
    "Insulin":                   (0,   900),
    "BMI":                       (10,  70),
    "DiabetesPedigreeFunction":  (0,   3),
    "Age":                       (1,   120),

    # Breast cancer features share a name pattern; validate generically below
}


def validate_patient_input(
    input_dict: dict[str, Any],
    feature_names: list[str],
) -> dict[str, float]:
    """
    Validate and coerce a raw patient input dictionary.

    Parameters
    ----------
    input_dict   : raw data from API / form submission
    feature_names: expected feature list from the trained model

    Returns
    -------
    Cleaned dict with float values.

    Raises
    ------
    ValidationError with a descriptive message on any failure.
    """
    errors: list[str] = []

    # ── 1. Check for missing required features ────────────────────────────────
    missing = [f for f in feature_names if f not in input_dict]
    if missing:
        errors.append(f"Missing required fields: {missing}")

    # ── 2. Type coercion + NaN check ─────────────────────────────────────────
    cleaned: dict[str, float] = {}
    for feat in feature_names:
        if feat not in input_dict:
            continue
        val = input_dict[feat]
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"'{feat}' must be numeric; got '{val}'")
            continue

        import math
        if math.isnan(val) or math.isinf(val):
            errors.append(f"'{feat}' contains NaN or Inf — provide a real value")
            continue

        cleaned[feat] = val

    # ── 3. Clinical range validation ──────────────────────────────────────────
    for feat, value in cleaned.items():
        if feat in FEATURE_BOUNDS:
            lo, hi = FEATURE_BOUNDS[feat]
            if not (lo <= value <= hi):
                errors.append(
                    f"'{feat}' value {value} is outside clinical range [{lo}, {hi}]"
                )

    if errors:
        msg = "; ".join(errors)
        logger.warning("Validation failed: %s", msg)
        raise ValidationError(msg)

    logger.debug("Input validated for features: %s", list(cleaned.keys()))
    return cleaned
