# 🏥 Disease Prediction System
### Production-ready end-to-end ML pipeline for clinical risk assessment

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-33%20passed-brightgreen)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27%2B-ff4b4b)](https://streamlit.io)

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Folder Structure](#-folder-structure)
4. [Datasets & Results](#-datasets--results)
5. [Patient Safety Fixes](#-patient-safety-fixes)
6. [Quick Start](#-quick-start)
7. [Training Pipeline](#-training-pipeline)
8. [Prediction & API](#-prediction--api)
9. [Streamlit Dashboard](#-streamlit-dashboard)
10. [Edge Cases Handled](#-edge-cases-handled)
11. [Evaluation Metrics](#-evaluation-metrics)
12. [Tech Stack](#-tech-stack)

---

## 🎯 Overview

A **production-grade machine learning system** that predicts the likelihood of three major diseases from structured patient data. Designed with clinical safety as the primary concern — not just accuracy.

**5 models trained per disease:**
- Logistic Regression, SVM, Random Forest, XGBoost *(GridSearchCV tuned)*
- Stacking Ensemble *(RF + XGB + SVM → Logistic Regression meta-learner)*

**Safety features built-in:**
- Optimal decision thresholds targeting 85–95% recall (not default 0.5)
- SMOTE oversampling for imbalanced datasets (training fold only — no leakage)
- SHAP explainability for every prediction
- Calibration curves to verify probability reliability
- PR-AUC as primary metric for imbalanced datasets
- Learning curve overfitting detection

---

## 🏗 Architecture

```
Patient Data (CSV / API / UI)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│  Load → Clean → Impute → Cap Outliers → Encode         │
│  Feature Engineering → Split (80/20) → Scale           │
│  SMOTE (training fold only, when imbalance > 1.3:1)    │
└──────────────────────┬─────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      TRAINING PIPELINE      │
        │  4 base classifiers         │
        │  GridSearchCV × 5-fold CV   │
        │  + Stacking Ensemble        │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │    SAFETY & EVALUATION      │
        │  Threshold optimisation     │
        │  Calibration check (ECE)    │
        │  PR curves + ROC curves     │
        │  SHAP explainability        │
        │  Learning curve diagnosis   │
        └──────┬────────────┬─────────┘
               │            │
   ┌───────────▼──┐   ┌─────▼──────────┐
   │  FastAPI REST│   │  Streamlit UI  │
   │  /predict/   │   │  Dashboard     │
   └──────────────┘   └────────────────┘
```

---

## 📁 Folder Structure

```
disease_prediction/
│
├── datasets/                         # Raw CSVs
│   ├── heart.csv
│   ├── diabetes.csv
│   └── breast_cancer.csv
│
├── src/
│   ├── config.py                     # Paths, hyperparams, dataset metadata
│   ├── data/
│   │   ├── data_loader.py            # Load, clean, impute, SMOTE, split, scale
│   │   └── eda.py                    # 5 EDA plot types per disease
│   ├── models/
│   │   ├── model_factory.py          # 4 GridSearchCV classifiers (regularised v2)
│   │   └── stacking_ensemble.py      # RF + XGB + SVM → LR meta-learner
│   ├── evaluation/
│   │   ├── evaluator.py              # Metrics, confusion matrix, ROC, calibration
│   │   ├── learning_curves.py        # Overfitting diagnosis and gap plots
│   │   ├── pr_curves.py              # PR curves and threshold trade-off plots
│   │   └── shap_explainer.py         # Summary, bar, waterfall, dependence plots
│   └── utils/
│       ├── logger.py                 # Rotating file + console logger
│       ├── model_persistence.py      # Save/load .joblib + .meta.json
│       ├── validators.py             # Clinical range validation
│       ├── threshold_optimizer.py    # Sweep thresholds, target recall, persist
│       └── calibrator.py            # Platt scaling + ECE measurement
│
├── trained_models/                   # Persisted artefacts
│   ├── heart/                        # *.joblib + *.meta.json
│   ├── diabetes/
│   └── breast_cancer/
│
├── reports/
│   ├── figures/                      # All PNG plots
│   │   ├── heart/
│   │   ├── diabetes/
│   │   └── breast_cancer/
│   ├── all_diseases_metrics.csv
│   ├── threshold_improvements.csv
│   └── calibration_report.csv
│
├── deployment/
│   ├── api/main.py                   # FastAPI REST API
│   └── frontend/app.py               # Streamlit dashboard
│
├── tests/
│   └── test_pipeline.py              # 33 unit + integration tests
│
├── train.py                          # Master training CLI
├── predict.py                        # Inference CLI
├── fix_patient_safety.py             # Standalone safety fixes
└── requirements.txt
```

---

## 📊 Datasets & Results

### ❤️ Heart Disease — 1,025 rows × 13 features

| Model | Accuracy | Recall | ROC-AUC | Opt Threshold |
|-------|----------|--------|---------|---------------|
| Random Forest | 1.000 | 1.000 | 1.000 | 0.25 |
| XGBoost | 1.000 | 1.000 | 1.000 | 0.10 |
| SVM | 0.995 | 0.991 | 0.9997 | 0.45 |
| Logistic Regression | 0.834 | 0.924 | 0.938 | 0.53 |

### 🩸 Diabetes — 768 rows × 10 features*

| Model | Default Recall | Optimised Recall | Threshold | FN Prevented |
|-------|---------------|-----------------|-----------|-------------|
| Random Forest | 0.500 | **0.907** | 0.225 | **22 patients** |
| Logistic Regression | 0.519 | **0.870** | 0.250 | **19 patients** |
| SVM | 0.481 | **0.852** | 0.275 | **18 patients** |
| XGBoost | 0.796 | **0.907** | 0.350 | **6 patients** |

*10 features after engineering: BMI_category + Glucose_Insulin

### 🔬 Breast Cancer — 569 rows × 30 features

| Model | Default Recall | Optimised Recall | Threshold | FN Prevented |
|-------|---------------|-----------------|-----------|-------------|
| Random Forest | 0.929 | **1.000** | 0.175 | **3 patients** |
| SVM | 0.952 | **0.976** | 0.375 | **2 patients** |
| XGBoost | 0.929 | **0.976** | 0.100 | **2 patients** |
| Logistic Regression | 0.952 | **0.976** | 0.300 | **1 patient** |

---

## 🛡 Patient Safety Fixes

Four clinical safety improvements beyond baseline accuracy:

**1. Threshold Tuning** — The default 0.5 threshold optimises accuracy, not patient
safety. Swept 0.10→0.75 and persisted the threshold that achieves ≥85% recall
(diabetes) / ≥92% (heart) / ≥95% (breast cancer). API and frontend use it automatically.
*Result: up to 22 fewer missed diabetics per 154 test patients.*

**2. Overfitting Fix** — Heart RF had unlimited depth, train AUC = 1.000 at all
training sizes. Fixed with `max_depth=8, min_samples_leaf=2`. Max train/val gap:
0.056 → 0.012. XGBoost regularised with `min_child_weight` + `reg_lambda`.

**3. Calibration Measurement** — XGBoost diabetes ECE = 0.086 (overconfident).
All models measured with `calibration_curve`. ECE stored in reports.
`CalibratedClassifierCV` wrapper available via `calibrator.py`.

**4. PR Curves** — PR-AUC added alongside ROC-AUC. On imbalanced datasets
ROC looks great even when recall is poor. PR-AUC shows the true cost.

---

## ⚡ Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train all diseases (EDA + SHAP + thresholds included)
python train.py

# 3. Run tests
python -m pytest tests/ -v

# 4. REST API
uvicorn deployment.api.main:app --reload --port 8000
# Docs → http://localhost:8000/docs

# 5. Dashboard
streamlit run deployment/frontend/app.py
# Opens → http://localhost:8501
```

---

## 🔁 Training Pipeline

```bash
python train.py                        # all diseases, full pipeline
python train.py --disease diabetes     # single disease
python train.py --no-shap              # skip SHAP (faster)
python train.py --no-smote             # disable SMOTE
python train.py --no-eda               # skip EDA plots
python train.py --no-save              # dry-run
```

| Step | What happens |
|------|-------------|
| Load | Read CSV, validate schema |
| Clean | Drop id / Unnamed columns |
| Impute | Zeros → NaN → median fill (diabetes physiological zeros) |
| Outliers | IQR Winsorization (1.5×) |
| Encode | Target → {0, 1} |
| Engineer | diabetes: BMI_category, Glucose_Insulin |
| Split | Stratified 80/20 |
| Scale | StandardScaler fit on train only |
| SMOTE | Train fold only when imbalance > 1.3:1 |
| EDA | Class distribution, heatmap, distributions, boxplots, pairplot |
| Train | 4 base models + stacking ensemble, GridSearchCV × 5-fold CV |
| Evaluate | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Plots | ROC, PR, calibration, feature importance, threshold trade-off |
| SHAP | Summary, bar, waterfall (highest-risk patient), dependence |
| Threshold | Sweep, persist optimal to meta.json |
| Save | .joblib bundle + .meta.json per model |
| Report | CSV per disease + combined summary |

---

## 🔌 Prediction & API

### Single patient (CLI)
```bash
python predict.py --disease heart --model "Random Forest"
```

### Batch CSV (CLI)
```bash
python predict.py --disease diabetes --model XGBoost \
  --input patients.csv --output results.csv
```

### REST API
```bash
# POST /predict/diabetes
curl -X POST http://localhost:8000/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "XGBoost",
    "features": {
      "Pregnancies":3, "Glucose":168, "BloodPressure":74,
      "SkinThickness":28, "Insulin":180, "BMI":33.7,
      "DiabetesPedigreeFunction":0.537, "Age":43
    }
  }'
```

**Response:**
```json
{
  "disease": "diabetes",
  "model_used": "XGBoost",
  "prediction": 1,
  "label": "Diabetic",
  "probability": 0.8312,
  "risk_level": "High",
  "threshold_used": 0.35,
  "latency_ms": 4.1
}
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/diseases` | List diseases |
| GET | `/diseases/{disease}/models` | Models + metrics + thresholds |
| POST | `/predict/{disease}` | Single patient |
| POST | `/predict/{disease}/batch` | Batch ≤ 500 |

---

## 🖥 Streamlit Dashboard

```bash
streamlit run deployment/frontend/app.py
```

Three tabs: **Predict** / **EDA Plots** / **SHAP Plots**

- Disease + model selector in sidebar
- Performance metrics panel
- Optimal threshold display with FN reduction info
- Interactive patient input (sliders + dropdowns)
- Live probability gauge + bar chart
- Risk level: Low / Moderate / High
- In-app EDA and SHAP plot viewer

---

## 🛡 Edge Cases Handled

| Challenge | Solution |
|-----------|----------|
| Physiological zeros | Replaced with NaN before imputation |
| Missing values | Median/mode imputation, fit on train only |
| Outliers | IQR Winsorization, no row deletion |
| Class imbalance | SMOTE + class_weight + scale_pos_weight |
| Data leakage | Scaler and SMOTE fit inside training fold only |
| Low recall at 0.5 | Per-disease optimal threshold persisted and auto-applied |
| Overfitting (heart RF) | max_depth, min_samples_leaf, reg_lambda in grid |
| Poor calibration | ECE measured, CalibratedClassifierCV available |
| Invalid API inputs | Clinical range checks, type coercion, NaN/Inf rejection |
| Corrupt model files | Metadata validation on load |

---

## 📐 Evaluation Metrics

| Metric | Healthcare importance |
|--------|-----------------------|
| **Recall** | Catches real cases — low recall = missed diagnoses |
| **Precision** | Reduces false alarms — unnecessary treatment/cost |
| **F1-Score** | Balance of Precision and Recall |
| **ROC-AUC** | Threshold-independent model ranking |
| **PR-AUC** | Better than ROC for imbalanced data |
| **ECE** | Are predicted probabilities trustworthy? |
| **Confusion Matrix** | Full TP/TN/FP/FN breakdown |

> In screening tools: **Recall > Precision**. Better to flag a healthy
> patient for follow-up (FP) than discharge a sick one (FN).

---

## 🧪 Tests — 33 Passing

```
TestDataLoader          10  shape, imputation, SMOTE, engineering, leakage
TestValidators           5  valid input, missing, type error, range, NaN
TestModelPersistence     6  list, load, inference, proba, threshold persisted
TestThresholdOptimizer   4  recall target, range, gain, FN reduction
TestCalibrator           2  ECE range, bin structure
TestStackingEnsemble     2  build, fit+predict
TestEndToEnd             4  full pipeline, threshold application
```

---

## 🧰 Tech Stack

| Category | Libraries |
|----------|-----------|
| Core ML | scikit-learn, xgboost, numpy, pandas |
| Imbalanced data | imbalanced-learn (SMOTE) |
| Explainability | shap |
| Visualisation | matplotlib, seaborn, plotly |
| API | fastapi, uvicorn, pydantic |
| Frontend | streamlit |
| Persistence | joblib |
| Testing | pytest |
| Logging | Python logging (rotating file) |

---

## ⚠️ Medical Disclaimer

> This system is intended **solely for research and educational purposes**.
> It does not constitute medical advice and must not replace consultation
> with a qualified healthcare professional.
