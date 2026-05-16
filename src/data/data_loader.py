"""
data/data_loader.py
--------------------
Handles loading, cleaning, imputation, outlier capping, feature
engineering, train/test splitting, scaling, and optional SMOTE.

SMOTE (Synthetic Minority Over-sampling Technique)
---------------------------------------------------
Applied ONLY inside the training fold after the train/test split.
Applying before splitting would cause data leakage (synthetic test
samples that the model has effectively "seen").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_CONFIG, DATASET_PATHS, RANDOM_STATE, TRAIN_TEST_SPLIT
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    End-to-end data loading and preprocessing for one disease dataset.

    Parameters
    ----------
    disease    : str  — "heart" | "diabetes" | "breast_cancer"
    use_smote  : bool — apply SMOTE to training split (default True for
                        imbalanced datasets, auto-disabled when balanced)
    """

    def __init__(self, disease: str, use_smote: bool = True) -> None:
        if disease not in DATASET_CONFIG:
            raise ValueError(
                f"Unknown disease '{disease}'. Choose from: {list(DATASET_CONFIG)}"
            )
        self.disease   = disease
        self.cfg       = DATASET_CONFIG[disease]
        self.use_smote = use_smote
        self.scaler    = StandardScaler()

        self.raw_df:       pd.DataFrame | None = None
        self.clean_df:     pd.DataFrame | None = None
        self.X_train:      np.ndarray | None   = None
        self.X_test:       np.ndarray | None   = None
        self.y_train:      np.ndarray | None   = None
        self.y_test:       np.ndarray | None   = None
        self.feature_names: list[str]          = []
        self.smote_applied: bool               = False

    # ── Public ────────────────────────────────────────────────────────────────

    def load_and_prepare(self) -> "DataLoader":
        logger.info("[%s] Starting data pipeline …", self.disease.upper())
        self._load()
        self._drop_unwanted_columns()
        self._handle_zero_as_missing()
        self._handle_missing_values()
        self._cap_outliers()
        self._encode_target()
        self._feature_engineering()
        self._split_and_scale()
        self._apply_smote()
        logger.info(
            "[%s] Ready — train: %d | test: %d | features: %d | smote: %s",
            self.disease.upper(), len(self.X_train), len(self.X_test),
            self.X_train.shape[1], self.smote_applied,
        )
        return self

    def get_data(self):
        if self.X_train is None:
            raise RuntimeError("Call load_and_prepare() first.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_class_names(self) -> list[str]:
        return self.cfg["class_names"]

    def preprocess_single(self, input_dict: dict) -> np.ndarray:
        """Scale a single patient dict into a model-ready array."""
        if not hasattr(self.scaler, "mean_"):
            raise RuntimeError("Scaler not fitted. Call load_and_prepare() first.")
        row = pd.DataFrame([input_dict])
        for col in self.cfg["zero_as_missing"]:
            if col in row.columns:
                row[col] = row[col].replace(0, np.nan)
        row = row.fillna(0)
        if self.disease == "diabetes":
            bmi = float(row.get("BMI", pd.Series([25]))[0])
            row["BMI_category"]    = next(l for e, l in zip([18.5,25,30,np.inf],[0,1,2,3]) if bmi < e)
            row["Glucose_Insulin"] = row.get("Glucose", 0) * row.get("Insulin", 0)
        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0.0
        return self.scaler.transform(row[self.feature_names].values.astype(np.float32))

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        path = DATASET_PATHS[self.disease]
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        self.raw_df = pd.read_csv(path)
        logger.info("  Loaded %s — shape: %s", Path(path).name, self.raw_df.shape)

    def _drop_unwanted_columns(self) -> None:
        cols = [c for c in self.cfg["drop_columns"] if c in self.raw_df.columns]
        if cols:
            self.raw_df = self.raw_df.drop(columns=cols)

    def _handle_zero_as_missing(self) -> None:
        for col in self.cfg["zero_as_missing"]:
            if col in self.raw_df.columns:
                n = (self.raw_df[col] == 0).sum()
                if n:
                    self.raw_df[col] = self.raw_df[col].replace(0, np.nan)
                    logger.debug("  %s: replaced %d zeros → NaN", col, n)

    def _handle_missing_values(self) -> None:
        target = self.cfg["target_column"]
        fcols  = [c for c in self.raw_df.columns if c != target]
        n_miss = self.raw_df[fcols].isnull().sum().sum()
        if n_miss:
            logger.info("  Imputing %d missing values …", n_miss)
            for col in fcols:
                if self.raw_df[col].isnull().any():
                    fill = (self.raw_df[col].median()
                            if self.raw_df[col].dtype in [np.float64, np.int64]
                            else self.raw_df[col].mode()[0])
                    self.raw_df[col] = self.raw_df[col].fillna(fill)

    def _cap_outliers(self, factor: float = 1.5) -> None:
        target   = self.cfg["target_column"]
        num_cols = [c for c in self.raw_df.select_dtypes(include=np.number).columns
                    if c != target]
        for col in num_cols:
            q1, q3 = self.raw_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
            self.raw_df[col] = self.raw_df[col].clip(lo, hi)

    def _encode_target(self) -> None:
        target    = self.cfg["target_column"]
        label_map = self.cfg["label_map"]
        if self.raw_df[target].dtype == object or isinstance(list(label_map.keys())[0], str):
            self.raw_df[target] = self.raw_df[target].map(label_map)
        self.cfg["label_map_inv"] = {v: self.cfg["class_names"][v] for v in [0, 1]}

    def _feature_engineering(self) -> None:
        if self.disease == "diabetes":
            bins, labels = [0, 18.5, 25, 30, np.inf], [0, 1, 2, 3]
            self.raw_df["BMI_category"] = pd.cut(
                self.raw_df["BMI"], bins=bins, labels=labels
            ).astype(int)
            self.raw_df["Glucose_Insulin"] = (
                self.raw_df["Glucose"] * self.raw_df["Insulin"]
            )
            logger.debug("  Diabetes: added BMI_category + Glucose_Insulin")
        self.clean_df = self.raw_df.copy()

    def _split_and_scale(self) -> None:
        target = self.cfg["target_column"]
        X = self.raw_df.drop(columns=[target]).values.astype(np.float32)
        y = self.raw_df[target].values.astype(int)
        self.feature_names = [c for c in self.raw_df.columns if c != target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.transform(self.X_test)

    def _apply_smote(self) -> None:
        """
        Apply SMOTE to the training split only.
        Auto-skips if:
          • use_smote=False
          • imbalance ratio < 1.3:1 (already balanced enough)
          • fewer than 50 minority samples (too few to synthesise safely)
        """
        if not self.use_smote:
            return
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        ratio = n_neg / max(n_pos, 1)
        if ratio < 1.3:
            logger.debug("  SMOTE skipped — ratio %.2f:1 (balanced enough)", ratio)
            return
        if min(n_neg, n_pos) < 50:
            logger.warning("  SMOTE skipped — minority class too small (%d samples)", min(n_neg, n_pos))
            return
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=RANDOM_STATE)
            self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
            self.smote_applied = True
            logger.info(
                "  SMOTE applied — train size: %d → %d (ratio was %.2f:1)",
                n_neg + n_pos, len(self.y_train), ratio,
            )
        except ImportError:
            logger.warning("  imbalanced-learn not installed — SMOTE skipped")
