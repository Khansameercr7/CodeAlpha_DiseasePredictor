"""
models/model_factory.py  (v2 — overfitting fixes)
---------------------------------------------------
Changes from v1
---------------
* Random Forest: added constrained depth variants (max_depth 5, 8, 12)
  to the grid so GridSearchCV can choose a regularised tree rather than
  always going unlimited depth.
* XGBoost: added min_child_weight and reg_lambda (L2 regularisation)
  to the grid — key levers against overfitting on small datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CV_FOLDS, RANDOM_STATE, SCORING_METRIC
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    Builds optimised sklearn-compatible classifiers via GridSearchCV.

    Parameters
    ----------
    class_ratio : float  — len(y==0) / len(y==1)
    n_samples   : int    — training set size (scales regularisation grid)
    """

    def __init__(self, class_ratio: float = 1.0, n_samples: int = 500) -> None:
        self.class_ratio = class_ratio
        self.n_samples   = n_samples
        self.cv = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )

    def get_all_models(self) -> dict[str, GridSearchCV]:
        return {
            "Logistic Regression": self._build_logistic_regression(),
            "SVM":                 self._build_svm(),
            "Random Forest":       self._build_random_forest(),
            "XGBoost":             self._build_xgboost(),
        }

    def _build_logistic_regression(self) -> GridSearchCV:
        return GridSearchCV(
            LogisticRegression(random_state=RANDOM_STATE),
            param_grid={
                "C":            [0.001, 0.01, 0.1, 1, 10, 100],
                "solver":       ["lbfgs", "liblinear"],
                "max_iter":     [1000],
                "class_weight": ["balanced", None],
            },
            cv=self.cv, scoring=SCORING_METRIC, n_jobs=-1, refit=True,
        )

    def _build_svm(self) -> GridSearchCV:
        return GridSearchCV(
            SVC(probability=True, random_state=RANDOM_STATE),
            param_grid={
                "C":            [0.1, 1, 10],
                "kernel":       ["rbf", "linear"],
                "gamma":        ["scale", "auto"],
                "class_weight": ["balanced", None],
            },
            cv=self.cv, scoring=SCORING_METRIC, n_jobs=-1, refit=True,
        )

    def _build_random_forest(self) -> GridSearchCV:
        """v2 FIX: constrained max_depth + tighter min_samples to prevent memorisation."""
        min_splits = [2, 5, 10] if self.n_samples < 1000 else [2, 5]
        min_leaf   = [1, 2, 4]  if self.n_samples < 1000 else [1, 2]
        return GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid={
                "n_estimators":      [100, 200, 300],
                "max_depth":         [5, 8, 12, None],   # KEY FIX: constrained depths
                "min_samples_split": min_splits,
                "min_samples_leaf":  min_leaf,
                "max_features":      ["sqrt", 0.5],
                "class_weight":      ["balanced", None],
            },
            cv=self.cv, scoring=SCORING_METRIC, n_jobs=-1, refit=True,
        )

    def _build_xgboost(self) -> GridSearchCV:
        """v2 FIX: min_child_weight + reg_lambda (L2) added to regularise small-dataset fits."""
        spw = max(1.0, round(self.class_ratio, 2))
        return GridSearchCV(
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                scale_pos_weight=spw,
                verbosity=0,
            ),
            param_grid={
                "n_estimators":     [100, 200, 300],
                "max_depth":        [3, 5, 7],
                "learning_rate":    [0.01, 0.05, 0.1],
                "subsample":        [0.7, 0.85, 1.0],
                "min_child_weight": [1, 5, 10],    # KEY FIX
                "reg_lambda":       [1, 5, 10],    # KEY FIX
            },
            cv=self.cv, scoring=SCORING_METRIC, n_jobs=-1, refit=True,
        )
