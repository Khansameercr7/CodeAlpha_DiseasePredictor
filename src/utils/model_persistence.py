"""
utils/model_persistence.py
---------------------------
Safe model saving and loading with metadata validation.
Prevents loading stale / incompatible model files at inference time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelPersistence:
    """
    Handles saving and loading of trained models + their metadata.

    Saves two artefacts per model:
        <disease>/<model_name>.joblib   — the fitted estimator
        <disease>/<model_name>.meta.json — metrics, feature names, timestamp
    """

    @staticmethod
    def save(
        model: Any,
        disease: str,
        model_name: str,
        scaler,
        feature_names: list[str],
        metrics: dict,
        class_names: list[str],
    ) -> Path:
        """
        Persist a trained model and its metadata to disk.

        Returns
        -------
        Path to the saved .joblib file.
        """
        save_dir = MODELS_DIR / disease
        save_dir.mkdir(parents=True, exist_ok=True)

        slug = model_name.lower().replace(" ", "_")
        model_path = save_dir / f"{slug}.joblib"
        meta_path  = save_dir / f"{slug}.meta.json"

        # Bundle model + scaler into one artefact for atomic deployment
        bundle = {
            "model":   model,
            "scaler":  scaler,
        }

        try:
            joblib.dump(bundle, model_path, compress=3)
            logger.info("  Model saved → %s", model_path)
        except Exception as exc:
            logger.error("  Failed to save model: %s", exc)
            raise

        # Serialisable subset of metrics
        safe_metrics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
            if k not in ("conf_matrix", "y_proba", "y_pred", "report")
        }

        meta = {
            "disease":       disease,
            "model_name":    model_name,
            "feature_names": feature_names,
            "class_names":   class_names,
            "metrics":       safe_metrics,
            "saved_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.debug("  Metadata saved → %s", meta_path)

        return model_path

    @staticmethod
    def load(disease: str, model_name: str) -> tuple[Any, Any, dict]:
        """
        Load a persisted model bundle + metadata.

        Returns
        -------
        (model, scaler, meta_dict)

        Raises
        ------
        FileNotFoundError  if .joblib or .meta.json are missing.
        ValueError         if metadata is corrupt or incomplete.
        """
        slug       = model_name.lower().replace(" ", "_")
        model_path = MODELS_DIR / disease / f"{slug}.joblib"
        meta_path  = MODELS_DIR / disease / f"{slug}.meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        try:
            bundle = joblib.load(model_path)
            model  = bundle["model"]
            scaler = bundle["scaler"]
        except Exception as exc:
            logger.error("  Failed to load model: %s", exc)
            raise

        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt metadata file: {meta_path}") from exc

        required_keys = {"disease", "model_name", "feature_names", "class_names"}
        missing = required_keys - meta.keys()
        if missing:
            raise ValueError(f"Metadata missing keys: {missing}")

        logger.info("  Loaded %s / %s  (saved: %s)", disease, model_name, meta.get("saved_at", "?"))
        return model, scaler, meta

    @staticmethod
    def list_saved(disease: str) -> list[str]:
        """Return names of all saved models for a disease."""
        save_dir = MODELS_DIR / disease
        if not save_dir.exists():
            return []
        return [p.stem for p in save_dir.glob("*.joblib")]
