"""
deployment/api/main.py
-----------------------
FastAPI backend — uses optimal_threshold from model metadata when available.

Run:  uvicorn deployment.api.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("pip install fastapi uvicorn pydantic")

import numpy as np

from config import DATASET_CONFIG
from utils.logger import get_logger
from utils.model_persistence import ModelPersistence
from utils.validators import ValidationError, validate_patient_input

logger = get_logger("api")

app = FastAPI(
    title       = "Disease Prediction API",
    description = "ML-powered clinical risk assessment — Heart Disease, Diabetes, Breast Cancer",
    version     = "2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET","POST"], allow_headers=["*"])

_cache: dict[str, Any] = {}


def _get(disease: str, model_name: str) -> dict:
    key = f"{disease}::{model_name}"
    if key not in _cache:
        try:
            model, scaler, meta = ModelPersistence.load(disease, model_name)
            _cache[key] = {"model": model, "scaler": scaler, "meta": meta}
        except FileNotFoundError:
            raise HTTPException(404, f"No saved model: disease='{disease}' model='{model_name}'. Run train.py first.")
    return _cache[key]


def _risk_level(p: float) -> str:
    return "Low" if p < 0.35 else ("Moderate" if p < 0.65 else "High")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_name: str = Field(default="Random Forest")
    features:   dict[str, float] = Field(...)

class PredictResponse(BaseModel):
    disease: str;    model_used: str;  prediction: int
    label: str;      probability: float;  risk_level: str
    threshold_used: float;  latency_ms: float

class BatchRequest(BaseModel):
    model_name: str = "Random Forest"
    patients:   list[dict[str, float]]

class HealthResponse(BaseModel):
    status: str; version: str; timestamp: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(status="healthy", version="2.0.0",
                          timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))


@app.get("/diseases", tags=["Meta"])
async def list_diseases():
    return {d: {"class_names": cfg["class_names"],
                "positive_label": cfg["positive_label"]}
            for d, cfg in DATASET_CONFIG.items()}


@app.get("/diseases/{disease}/models", tags=["Meta"])
async def list_models(disease: str):
    if disease not in DATASET_CONFIG:
        raise HTTPException(404, f"Unknown disease: {disease}")
    saved = ModelPersistence.list_saved(disease)
    out = []
    for slug in saved:
        mname = slug.replace("_", " ").title()
        try:
            _, _, meta = ModelPersistence.load(disease, mname)
            out.append({
                "model_name":       mname,
                "metrics":          meta.get("metrics", {}),
                "optimal_threshold": meta.get("optimal_threshold", 0.5),
                "feature_count":    len(meta.get("feature_names", [])),
                "saved_at":         meta.get("saved_at", ""),
            })
        except Exception:
            pass
    return out


@app.post("/predict/{disease}", response_model=PredictResponse, tags=["Prediction"])
async def predict(disease: str, req: PredictRequest):
    if disease not in DATASET_CONFIG:
        raise HTTPException(404, f"Unknown disease: {disease}")
    bundle = _get(disease, req.model_name)
    model, scaler, meta = bundle["model"], bundle["scaler"], bundle["meta"]

    try:
        cleaned = validate_patient_input(req.features, meta["feature_names"])
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    row   = np.array([[cleaned.get(f, 0.0) for f in meta["feature_names"]]], dtype=np.float32)
    X     = scaler.transform(row)
    t0    = time.perf_counter()
    proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else 0.5

    # Use persisted optimal threshold if available
    threshold = float(meta.get("optimal_threshold", 0.5))
    pred      = int(proba >= threshold)
    latency   = (time.perf_counter() - t0) * 1000

    logger.info("Predict [%s/%s] p=%.3f t=%.3f → %s  (%.1fms)",
                disease, req.model_name, proba, threshold,
                meta["class_names"][pred], latency)

    return PredictResponse(
        disease        = disease,
        model_used     = req.model_name,
        prediction     = pred,
        label          = meta["class_names"][pred],
        probability    = round(proba, 4),
        risk_level     = _risk_level(proba),
        threshold_used = threshold,
        latency_ms     = round(latency, 2),
    )


@app.post("/predict/{disease}/batch", tags=["Prediction"])
async def predict_batch(disease: str, req: BatchRequest):
    if disease not in DATASET_CONFIG:
        raise HTTPException(404, f"Unknown disease: {disease}")
    if len(req.patients) > 500:
        raise HTTPException(413, "Batch size must be ≤ 500")

    bundle = _get(disease, req.model_name)
    model, scaler, meta = bundle["model"], bundle["scaler"], bundle["meta"]
    threshold = float(meta.get("optimal_threshold", 0.5))
    results   = []

    for i, patient in enumerate(req.patients):
        try:
            cleaned = validate_patient_input(patient, meta["feature_names"])
            row     = np.array([[cleaned.get(f, 0.0) for f in meta["feature_names"]]], dtype=np.float32)
            X       = scaler.transform(row)
            proba   = float(model.predict_proba(X)[0][1])
            pred    = int(proba >= threshold)
            results.append({"index": i, "prediction": pred,
                             "label": meta["class_names"][pred],
                             "probability": round(proba, 4)})
        except ValidationError as exc:
            results.append({"index": i, "error": str(exc)})

    return {"disease": disease, "model_used": req.model_name,
            "threshold_used": threshold, "count": len(results), "results": results}
