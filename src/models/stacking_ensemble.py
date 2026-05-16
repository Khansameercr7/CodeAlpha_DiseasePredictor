"""
models/stacking_ensemble.py
-----------------------------
Stacking ensemble: RF + XGBoost + SVM base learners,
Logistic Regression meta-learner.

Why stacking works
------------------
Each base model captures different decision boundaries.
The meta-learner learns which base model to trust in which region —
typically yielding +1–4% AUC over the best single model.

Design
------
• Base learners use out-of-fold predictions (5-fold) so the
  meta-learner never trains on data the base models saw — no leakage.
• Final base models are refit on full training data.
• Passthrough=False: meta-learner sees only base model outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CV_FOLDS, RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)


def build_stacking_ensemble(class_ratio: float = 1.0) -> StackingClassifier:
    """
    Build a stacking ensemble for disease classification.

    Parameters
    ----------
    class_ratio : float — neg/pos ratio for XGBoost scale_pos_weight

    Returns
    -------
    sklearn StackingClassifier (not yet fitted)
    """
    spw = max(1.0, round(class_ratio, 2))
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    base_learners = [
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.85, min_child_weight=5, reg_lambda=5,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        )),
        ("svm", SVC(
            C=1, kernel="rbf", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ]

    meta_learner = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE
    )

    stack = StackingClassifier(
        estimators       = base_learners,
        final_estimator  = meta_learner,
        cv               = cv,
        stack_method     = "predict_proba",
        passthrough      = False,
        n_jobs           = -1,
    )

    logger.debug("Stacking ensemble built: RF + XGB + SVM → LR meta-learner")
    return stack
