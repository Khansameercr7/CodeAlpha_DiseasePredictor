"""
predict.py
----------
Standalone inference script — loads a saved model and predicts
disease risk for a single patient or a batch CSV file.

Usage
-----
    # Single patient (interactive prompts)
    python predict.py --disease heart --model "Random Forest"

    # Batch prediction from CSV
    python predict.py --disease diabetes --model XGBoost --input patients.csv --output results.csv

    # List available saved models
    python predict.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import DATASET_CONFIG
from utils.logger import get_logger
from utils.model_persistence import ModelPersistence
from utils.validators import ValidationError, validate_patient_input

logger = get_logger("predict")


# ─────────────────────────────────────────────────────────────────────────────
# Core predictor
# ─────────────────────────────────────────────────────────────────────────────

class DiseasePredictor:
    """
    Loads a trained model + scaler and exposes predict / predict_proba.

    Parameters
    ----------
    disease    : str  — "heart" | "diabetes" | "breast_cancer"
    model_name : str  — e.g. "Random Forest"
    """

    def __init__(self, disease: str, model_name: str) -> None:
        self.disease    = disease
        self.model_name = model_name
        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        self.model, self.scaler, self.meta = ModelPersistence.load(
            self.disease, self.model_name
        )
        self.feature_names = self.meta["feature_names"]
        self.class_names   = self.meta["class_names"]

    # ─────────────────────────────────────────────────────────────────────────
    def predict_single(self, patient_data: dict) -> dict:
        """
        Predict disease risk for a single patient.

        Parameters
        ----------
        patient_data : dict  key=feature_name, value=raw measurement

        Returns
        -------
        {
          "prediction":   int    (0 or 1),
          "label":        str    (human-readable),
          "probability":  float  (confidence for positive class),
          "risk_level":   str    ("Low" | "Moderate" | "High"),
        }
        """
        # Validate input
        cleaned = validate_patient_input(patient_data, self.feature_names)

        # Build feature vector (ordered as model expects)
        row = np.array([[cleaned.get(f, 0.0) for f in self.feature_names]], dtype=np.float32)
        X   = self.scaler.transform(row)

        pred  = int(self.model.predict(X)[0])
        proba = (
            float(self.model.predict_proba(X)[0][1])
            if hasattr(self.model, "predict_proba")
            else float(pred)
        )

        return {
            "prediction":  pred,
            "label":       self.class_names[pred],
            "probability": round(proba, 4),
            "risk_level":  _risk_level(proba),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for a DataFrame of patients.

        Skips rows that fail validation and marks them with an error flag.
        """
        results = []
        for idx, row in df.iterrows():
            patient = row.to_dict()
            try:
                result = self.predict_single(patient)
            except ValidationError as exc:
                result = {
                    "prediction": -1,
                    "label":      "VALIDATION_ERROR",
                    "probability": None,
                    "risk_level":  str(exc),
                }
            result["row_index"] = idx
            results.append(result)

        return pd.DataFrame(results).set_index("row_index")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Moderate"
    else:
        return "High"


def _interactive_input(feature_names: list[str]) -> dict:
    """Prompt the user to enter each feature value interactively."""
    print("\nEnter patient measurements (press Enter to skip optional fields):\n")
    data = {}
    for feat in feature_names:
        while True:
            raw = input(f"  {feat}: ").strip()
            if raw == "":
                data[feat] = 0.0
                break
            try:
                data[feat] = float(raw)
                break
            except ValueError:
                print(f"    ✗  Please enter a numeric value for '{feat}'")
    return data


def _list_models() -> None:
    from config import DATASET_CONFIG
    print("\nSaved models:\n")
    for disease in DATASET_CONFIG:
        models = ModelPersistence.list_saved(disease)
        print(f"  {disease}: {models if models else '(none)'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Disease Prediction — Inference")
    parser.add_argument("--disease", choices=list(DATASET_CONFIG), help="Disease type")
    parser.add_argument("--model",   default="Random Forest",       help="Model name")
    parser.add_argument("--input",   help="Path to input CSV for batch prediction")
    parser.add_argument("--output",  default="predictions.csv",     help="Output CSV path")
    parser.add_argument("--list",    action="store_true",           help="List saved models")
    args = parser.parse_args()

    if args.list:
        _list_models()
        return

    if not args.disease:
        parser.error("--disease is required unless --list is used")

    predictor = DiseasePredictor(args.disease, args.model)

    if args.input:
        # ── Batch mode ────────────────────────────────────────────────────────
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)

        df = pd.read_csv(input_path)
        logger.info("Batch prediction for %d patients …", len(df))
        results = predictor.predict_batch(df)
        results.to_csv(args.output)
        logger.info("Results saved → %s", args.output)
        print(results.to_string())

    else:
        # ── Interactive single-patient mode ───────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  Disease Prediction: {args.disease.upper().replace('_', ' ')}")
        print(f"  Model:              {args.model}")
        print(f"{'─'*50}")

        patient_data = _interactive_input(predictor.feature_names)
        result = predictor.predict_single(patient_data)

        print(f"\n{'─'*50}")
        print(f"  RESULT")
        print(f"{'─'*50}")
        print(f"  Diagnosis   : {result['label']}")
        print(f"  Confidence  : {result['probability'] * 100:.1f}%")
        print(f"  Risk Level  : {result['risk_level']}")
        print(f"{'─'*50}\n")
        print("  ⚠  This tool is for research purposes only.")
        print("     Always consult a licensed medical professional.\n")


if __name__ == "__main__":
    main()
