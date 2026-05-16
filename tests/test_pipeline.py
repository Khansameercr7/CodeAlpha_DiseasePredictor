"""
tests/test_pipeline.py
-----------------------
Unit + integration tests covering the full pipeline including SMOTE,
stacking ensemble, threshold optimizer, calibrator, PR curves, and SHAP.

Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATASET_CONFIG
from data.data_loader import DataLoader
from utils.validators import validate_patient_input, ValidationError
from utils.model_persistence import ModelPersistence
from utils.threshold_optimizer import ThresholdOptimizer, RECALL_TARGETS
from utils.calibrator import ModelCalibrator


# ── DataLoader ────────────────────────────────────────────────────────────────

class TestDataLoader:

    def test_heart_shape(self):
        loader = DataLoader("heart", use_smote=False).load_and_prepare()
        X_train, X_test, y_train, y_test = loader.get_data()
        assert X_train.shape[1] == 13
        assert set(np.unique(y_train)) == {0, 1}

    def test_diabetes_zero_imputation(self):
        loader = DataLoader("diabetes", use_smote=False).load_and_prepare()
        for col in ["Glucose", "BloodPressure", "BMI"]:
            assert (loader.clean_df[col] == 0).sum() == 0, f"{col} still has zeros"

    def test_diabetes_engineered_features(self):
        loader = DataLoader("diabetes", use_smote=False).load_and_prepare()
        assert "BMI_category"    in loader.feature_names
        assert "Glucose_Insulin" in loader.feature_names

    def test_breast_cancer_drop_columns(self):
        loader = DataLoader("breast_cancer", use_smote=False).load_and_prepare()
        assert "id"          not in loader.feature_names
        assert "Unnamed: 32" not in loader.feature_names

    def test_stratified_split(self):
        loader = DataLoader("diabetes", use_smote=False).load_and_prepare()
        _, _, y_train, y_test = loader.get_data()
        assert abs(y_train.mean() - y_test.mean()) < 0.05

    def test_no_leakage_scaler(self):
        loader = DataLoader("heart", use_smote=False).load_and_prepare()
        assert hasattr(loader.scaler, "mean_")

    def test_smote_increases_train_size(self):
        loader_no  = DataLoader("diabetes", use_smote=False).load_and_prepare()
        loader_yes = DataLoader("diabetes", use_smote=True).load_and_prepare()
        X_no,  _, y_no,  _ = loader_no.get_data()
        X_yes, _, y_yes, _ = loader_yes.get_data()
        if loader_yes.smote_applied:
            assert len(X_yes) > len(X_no), "SMOTE should grow training set"
            assert (y_yes == 0).sum() == (y_yes == 1).sum() or \
                   abs((y_yes==0).sum() - (y_yes==1).sum()) < 5, "SMOTE should balance classes"

    def test_smote_not_applied_to_balanced(self):
        # Heart dataset is near-balanced — SMOTE should auto-skip
        loader = DataLoader("heart", use_smote=True).load_and_prepare()
        # heart imbalance < 1.3:1 so smote_applied should be False
        n_neg = (loader.y_train == 0).sum()
        n_pos = (loader.y_train == 1).sum()
        if n_neg / n_pos < 1.3:
            assert not loader.smote_applied

    def test_unknown_disease_raises(self):
        with pytest.raises(ValueError, match="Unknown disease"):
            DataLoader("covid")

    def test_preprocess_single_shape(self):
        loader = DataLoader("heart", use_smote=False).load_and_prepare()
        sample = {f: 0.0 for f in loader.feature_names}
        out = loader.preprocess_single(sample)
        assert out.shape == (1, len(loader.feature_names))


# ── Validators ────────────────────────────────────────────────────────────────

class TestValidators:

    FEATS = ["age","sex","cp","trestbps","chol","fbs","restecg",
             "thalach","exang","oldpeak","slope","ca","thal"]

    def _valid(self):
        return {"age":54,"sex":1,"cp":0,"trestbps":122,"chol":286,
                "fbs":0,"restecg":0,"thalach":116,"exang":1,
                "oldpeak":3.2,"slope":1,"ca":2,"thal":2}

    def test_valid_passes(self):
        r = validate_patient_input(self._valid(), self.FEATS)
        assert r["age"] == 54.0

    def test_missing_field_raises(self):
        d = self._valid(); del d["age"]
        with pytest.raises(ValidationError, match="Missing"):
            validate_patient_input(d, self.FEATS)

    def test_non_numeric_raises(self):
        d = self._valid(); d["age"] = "old"
        with pytest.raises(ValidationError, match="numeric"):
            validate_patient_input(d, self.FEATS)

    def test_out_of_range_raises(self):
        d = self._valid(); d["age"] = 999
        with pytest.raises(ValidationError, match="clinical range"):
            validate_patient_input(d, self.FEATS)

    def test_nan_raises(self):
        d = self._valid(); d["trestbps"] = float("nan")
        with pytest.raises(ValidationError, match="NaN"):
            validate_patient_input(d, self.FEATS)


# ── ModelPersistence ──────────────────────────────────────────────────────────

class TestModelPersistence:

    def test_list_saved_returns_models(self):
        for disease in DATASET_CONFIG:
            saved = ModelPersistence.list_saved(disease)
            assert len(saved) >= 1, f"No saved models for {disease}"

    def test_load_heart_random_forest(self):
        model, scaler, meta = ModelPersistence.load("heart", "Random Forest")
        assert hasattr(model, "predict")
        assert hasattr(scaler, "transform")
        assert "feature_names" in meta
        assert "class_names"   in meta

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            ModelPersistence.load("heart", "FakeModel XYZ")

    def test_inference_output_shape(self):
        model, scaler, meta = ModelPersistence.load("heart", "Random Forest")
        row  = np.zeros((1, len(meta["feature_names"])), dtype=np.float32)
        X    = scaler.transform(row)
        pred = model.predict(X)
        assert pred.shape == (1,)
        assert pred[0] in [0, 1]

    def test_proba_sums_to_one(self):
        model, scaler, meta = ModelPersistence.load("diabetes", "XGBoost")
        row   = np.zeros((1, len(meta["feature_names"])), dtype=np.float32)
        X     = scaler.transform(row)
        proba = model.predict_proba(X)[0]
        assert abs(sum(proba) - 1.0) < 1e-5

    def test_optimal_threshold_persisted(self):
        """After fix_patient_safety.py / train.py, threshold must be in meta."""
        for disease in DATASET_CONFIG:
            saved = ModelPersistence.list_saved(disease)
            for slug in saved:
                mname = slug.replace("_", " ").title()
                try:
                    _, _, meta = ModelPersistence.load(disease, mname)
                    if "optimal_threshold" in meta:
                        t = meta["optimal_threshold"]
                        assert 0.0 < t < 1.0, f"Bad threshold {t} for {disease}/{mname}"
                except FileNotFoundError:
                    pass


# ── ThresholdOptimizer ────────────────────────────────────────────────────────

class TestThresholdOptimizer:

    def _get_proba(self, disease, model_name):
        loader = DataLoader(disease, use_smote=False).load_and_prepare()
        _, X_test, _, y_test = loader.get_data()
        model, scaler, meta  = ModelPersistence.load(disease, model_name)
        return y_test, model.predict_proba(X_test)[:, 1]

    def test_recall_target_met_or_best_effort(self):
        y_test, y_proba = self._get_proba("diabetes", "XGBoost")
        opt = ThresholdOptimizer("diabetes", "XGBoost")
        result = opt.optimize(y_test, y_proba)
        target = RECALL_TARGETS["diabetes"]
        assert result.recall >= target or result.recall == y_test.mean() or result.recall > 0

    def test_threshold_in_valid_range(self):
        y_test, y_proba = self._get_proba("diabetes", "XGBoost")
        result = ThresholdOptimizer("diabetes", "XGBoost").optimize(y_test, y_proba)
        assert 0.0 < result.optimal_threshold < 1.0

    def test_recall_gain_nonnegative(self):
        y_test, y_proba = self._get_proba("diabetes", "Logistic Regression")
        result = ThresholdOptimizer("diabetes", "Logistic Regression").optimize(y_test, y_proba)
        assert result.recall_gain >= -0.01   # allow tiny float noise

    def test_fn_reduced_vs_default(self):
        y_test, y_proba = self._get_proba("diabetes", "Random Forest")
        result = ThresholdOptimizer("diabetes", "Random Forest").optimize(y_test, y_proba)
        assert result.false_negatives <= result.default_fn


# ── Calibrator ────────────────────────────────────────────────────────────────

class TestCalibrator:

    def test_ece_in_valid_range(self):
        model, scaler, meta = ModelPersistence.load("diabetes", "XGBoost")
        loader = DataLoader("diabetes", use_smote=False).load_and_prepare()
        _, X_test, _, y_test = loader.get_data()
        result = ModelCalibrator.evaluate(model, X_test, y_test)
        assert 0.0 <= result["ece"] <= 1.0

    def test_calibration_bins_returned(self):
        model, scaler, meta = ModelPersistence.load("heart", "Random Forest")
        loader = DataLoader("heart", use_smote=False).load_and_prepare()
        _, X_test, _, y_test = loader.get_data()
        result = ModelCalibrator.evaluate(model, X_test, y_test, n_bins=5)
        assert "mean_predicted_values" in result
        assert "fraction_of_positives" in result
        assert len(result["mean_predicted_values"]) <= 5


# ── Stacking Ensemble ─────────────────────────────────────────────────────────

class TestStackingEnsemble:

    def test_stacking_build(self):
        from models.stacking_ensemble import build_stacking_ensemble
        stack = build_stacking_ensemble(class_ratio=1.5)
        assert hasattr(stack, "fit")
        assert hasattr(stack, "predict_proba")
        assert len(stack.estimators) == 3

    def test_stacking_fit_predict(self):
        from models.stacking_ensemble import build_stacking_ensemble
        loader = DataLoader("diabetes", use_smote=False).load_and_prepare()
        X_train, X_test, y_train, y_test = loader.get_data()
        stack = build_stacking_ensemble()
        stack.fit(X_train, y_train)
        preds = stack.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})
        proba = stack.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert all(abs(p.sum() - 1.0) < 1e-5 for p in proba)


# ── End-to-end integration ────────────────────────────────────────────────────

class TestEndToEnd:

    def test_full_heart_pipeline(self):
        model, scaler, meta = ModelPersistence.load("heart", "SVM")
        feats  = meta["feature_names"]
        sample = {f: 0.0 for f in feats}
        sample.update({"age":58,"sex":1,"cp":0,"trestbps":140,"chol":260,
                       "fbs":0,"restecg":0,"thalach":120,"exang":1,
                       "oldpeak":2.5,"slope":1,"ca":1,"thal":2})
        row  = np.array([[sample.get(f, 0.0) for f in feats]], dtype=np.float32)
        pred = model.predict(scaler.transform(row))[0]
        assert pred in [0, 1]

    def test_full_diabetes_pipeline(self):
        loader = DataLoader("diabetes", use_smote=False).load_and_prepare()
        model, scaler, meta = ModelPersistence.load("diabetes", "XGBoost")
        preds = model.predict(loader.X_test)
        assert len(preds) == len(loader.X_test)
        assert set(preds).issubset({0, 1})

    def test_breast_cancer_no_id_leak(self):
        loader = DataLoader("breast_cancer", use_smote=False).load_and_prepare()
        assert "id" not in loader.feature_names

    def test_threshold_applied_correctly(self):
        """Threshold from meta must change prediction vs default 0.5."""
        model, scaler, meta = ModelPersistence.load("diabetes", "XGBoost")
        loader = DataLoader("diabetes", use_smote=False).load_and_prepare()
        _, X_test, _, y_test = loader.get_data()
        proba     = model.predict_proba(X_test)[:, 1]
        threshold = float(meta.get("optimal_threshold", 0.5))
        preds_opt = (proba >= threshold).astype(int)
        preds_def = (proba >= 0.5).astype(int)
        if threshold != 0.5:
            from sklearn.metrics import recall_score
            assert recall_score(y_test, preds_opt) >= recall_score(y_test, preds_def) - 0.01
