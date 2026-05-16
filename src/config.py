"""
config.py
---------
Central configuration for the Disease Prediction System.
All paths, hyperparameters, and dataset meta-data live here.
"""

import os
from pathlib import Path

# ─── Root Paths ────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATASETS_DIR    = ROOT_DIR / "datasets"
MODELS_DIR      = ROOT_DIR / "trained_models"
REPORTS_DIR     = ROOT_DIR / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"
LOGS_DIR        = ROOT_DIR / "logs"

# ─── Dataset Paths ─────────────────────────────────────────────────────────────
DATASET_PATHS = {
    "heart":        DATASETS_DIR / "heart.csv",
    "diabetes":     DATASETS_DIR / "diabetes.csv",
    "breast_cancer": DATASETS_DIR / "breast_cancer.csv",
}

# ─── Dataset-level Configuration ───────────────────────────────────────────────
DATASET_CONFIG = {
    "heart": {
        "target_column": "target",
        "drop_columns":  [],
        "categorical_columns": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "zero_as_missing": [],          # no physiological zeros here
        "label_map": {0: "No Disease", 1: "Heart Disease"},
        "class_names": ["No Disease", "Heart Disease"],
        "positive_label": "Heart Disease",
    },
    "diabetes": {
        "target_column": "Outcome",
        "drop_columns":  [],
        "categorical_columns": [],
        # Glucose, BP, SkinThickness, Insulin, BMI cannot be 0 physiologically
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "label_map": {0: "Non-Diabetic", 1: "Diabetic"},
        "class_names": ["Non-Diabetic", "Diabetic"],
        "positive_label": "Diabetic",
    },
    "breast_cancer": {
        "target_column": "diagnosis",
        "drop_columns":  ["id", "Unnamed: 32"],
        "categorical_columns": [],
        "zero_as_missing": [],
        "label_map": {"B": 0, "M": 1},  # B=Benign, M=Malignant
        "class_names": ["Benign", "Malignant"],
        "positive_label": "Malignant",
    },
}

# ─── Model Hyperparameters ─────────────────────────────────────────────────────
MODEL_PARAMS = {
    "logistic_regression": {
        "C":          [0.001, 0.01, 0.1, 1, 10, 100],
        "solver":     ["lbfgs", "liblinear"],
        "max_iter":   [1000],
        "class_weight": ["balanced", None],
    },
    "svm": {
        "C":      [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
        "class_weight": ["balanced", None],
    },
    "random_forest": {
        "n_estimators":     [100, 200],
        "max_depth":        [None, 5, 10],
        "min_samples_split":[2, 5],
        "class_weight":     ["balanced", None],
    },
    "xgboost": {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample":     [0.8, 1.0],
        "scale_pos_weight": [1],        # adjusted per dataset in trainer
    },
}

# ─── Training Settings ─────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT  = 0.2
RANDOM_STATE      = 42
CV_FOLDS          = 5
SCORING_METRIC    = "roc_auc"   # primary optimization metric

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
