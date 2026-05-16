"""
deployment/frontend/app.py
---------------------------
Streamlit dashboard — v2 with optimal threshold, stacking ensemble,
calibration info, and SHAP plot viewer.

Run: streamlit run deployment/frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("pip install streamlit plotly")

from config import DATASET_CONFIG, FIGURES_DIR
from utils.model_persistence import ModelPersistence

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disease Prediction System", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-title{font-size:2.1rem;font-weight:700;color:#1a237e}
.subtitle{color:#546e7a;font-size:.95rem;margin-bottom:1rem}
.metric-card{background:linear-gradient(135deg,#e3f2fd,#fff);border-radius:10px;
  padding:.9rem 1.2rem;border-left:4px solid #1565c0;margin-bottom:.7rem;color:#1a237e}
.risk-Low{color:#2e7d32;font-weight:700;font-size:1.4rem}
.risk-Moderate{color:#e65100;font-weight:700;font-size:1.4rem}
.risk-High{color:#b71c1c;font-weight:700;font-size:1.4rem}
.disclaimer{background:#fff3e0;border-radius:8px;padding:.7rem 1rem;
  font-size:.82rem;color:#6d4c41;margin-top:1rem}
.thresh-info{background:#e8f5e9;border-radius:8px;padding:.5rem .9rem;
  font-size:.85rem;color:#2e7d32;margin-bottom:.5rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(disease, model_name):
    return ModelPersistence.load(disease, model_name)


def make_gauge(probability, label):
    color = "#2e7d32" if probability < 0.35 else ("#e65100" if probability < 0.65 else "#b71c1c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": f"Risk Probability<br><span style='font-size:13px'>{label}</span>"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": color},
            "steps": [{"range": [0, 35], "color": "#e8f5e9"},
                      {"range": [35, 65], "color": "#fff3e0"},
                      {"range": [65, 100], "color": "#ffebee"}],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=55, b=15))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Disease Predictor")
    st.markdown("---")
    disease_labels = {"heart": "❤️ Heart Disease",
                      "diabetes": "🩸 Diabetes",
                      "breast_cancer": "🔬 Breast Cancer"}
    disease = st.selectbox("Select Disease", list(disease_labels),
                           format_func=lambda d: disease_labels[d])

    saved = ModelPersistence.list_saved(disease)
    if not saved:
        st.error("No trained models found.\nRun `python train.py` first.")
        st.stop()

    model_map = {s: s.replace("_", " ").title() for s in saved}
    slug      = st.selectbox("Select Model", saved, format_func=lambda s: model_map[s])
    mname     = model_map[slug]

    st.markdown("---")
    st.caption("ML-powered screening — not a substitute for professional diagnosis.")


# ── Load ──────────────────────────────────────────────────────────────────────
try:
    model, scaler, meta = load_model(disease, mname)
except Exception as exc:
    st.error(f"Failed to load model: {exc}"); st.stop()

feature_names  = meta["feature_names"]
class_names    = meta["class_names"]
saved_metrics  = meta.get("metrics", {})
opt_threshold  = float(meta.get("optimal_threshold", 0.5))
thresh_metrics = meta.get("threshold_metrics", {})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f'<p class="main-title">🏥 Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Predicting <b>{disease_labels[disease]}</b> using <b>{mname}</b></p>',
            unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────────────────────────────────
with st.expander("📊 Model Performance Metrics", expanded=False):
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Accuracy","accuracy"),("Precision","precision"),
                                         ("Recall","recall"),("F1","f1"),("ROC-AUC","roc_auc")]):
        v = saved_metrics.get(key, "—")
        col.metric(label, f"{v:.3f}" if isinstance(v, float) else v)

    # Threshold info
    if thresh_metrics:
        st.markdown(
            f'<div class="thresh-info">⚡ Optimal threshold: <b>{opt_threshold:.3f}</b> '
            f'(default 0.50 → recall {thresh_metrics.get("default_recall",0):.3f} | '
            f'optimised → recall {thresh_metrics.get("recall",0):.3f} | '
            f'{thresh_metrics.get("default_fn",0) - thresh_metrics.get("false_negatives",0)} fewer missed patients)</div>',
            unsafe_allow_html=True
        )

# ── EDA / SHAP plots viewer ───────────────────────────────────────────────────
tabs = st.tabs(["🩺 Predict", "📈 EDA Plots", "🧠 SHAP Plots"])

with tabs[1]:
    fig_dir = FIGURES_DIR / disease
    plot_files = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if plot_files:
        sel = st.selectbox("Select plot", [f.name for f in plot_files])
        st.image(str(fig_dir / sel), use_container_width=True)
    else:
        st.info("Run `python train.py --eda` to generate EDA plots.")

with tabs[2]:
    shap_files = sorted(FIGURES_DIR.glob(f"{disease}/shap_*.png")) if (FIGURES_DIR / disease).exists() else []
    if shap_files:
        sel2 = st.selectbox("Select SHAP plot", [f.name for f in shap_files])
        st.image(str(FIGURES_DIR / disease / sel2), use_container_width=True)
    else:
        st.info("Run `python train.py` (without --no-shap) to generate SHAP plots.")

# ── Prediction tab ────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🩺 Enter Patient Data")

    input_data: dict = {}

    if disease == "heart":
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data["age"]      = st.slider("Age", 20, 100, 54)
            input_data["sex"]      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            input_data["cp"]       = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
            input_data["trestbps"] = st.slider("Resting BP (mmHg)", 80, 220, 130)
            input_data["chol"]     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        with c2:
            input_data["fbs"]      = st.selectbox("Fasting Blood Sugar >120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            input_data["restecg"]  = st.selectbox("Resting ECG (0–2)", [0,1,2])
            input_data["thalach"]  = st.slider("Max Heart Rate", 60, 220, 150)
            input_data["exang"]    = st.selectbox("Exercise Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c3:
            input_data["oldpeak"]  = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
            input_data["slope"]    = st.selectbox("ST Slope (0–2)", [0,1,2])
            input_data["ca"]       = st.selectbox("Major Vessels (0–4)", [0,1,2,3,4])
            input_data["thal"]     = st.selectbox("Thal (0=normal,1=fixed,2=reversible)", [0,1,2])

    elif disease == "diabetes":
        c1, c2 = st.columns(2)
        with c1:
            input_data["Pregnancies"]  = st.slider("Pregnancies", 0, 20, 2)
            input_data["Glucose"]      = st.slider("Glucose (mg/dl)", 40, 400, 120)
            input_data["BloodPressure"]= st.slider("Blood Pressure (mmHg)", 30, 200, 70)
            input_data["SkinThickness"]= st.slider("Skin Thickness (mm)", 0, 100, 20)
        with c2:
            input_data["Insulin"]      = st.slider("Insulin (μU/ml)", 0, 900, 80)
            input_data["BMI"]          = st.slider("BMI", 10.0, 70.0, 25.0, 0.1)
            input_data["DiabetesPedigreeFunction"] = st.slider("Pedigree Function", 0.0, 3.0, 0.5, 0.01)
            input_data["Age"]          = st.slider("Age", 10, 120, 35)

    elif disease == "breast_cancer":
        st.info("Enter nuclear feature measurements (0 = use model default).")
        chunks = [feature_names[i:i+4] for i in range(0, len(feature_names), 4)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for col, feat in zip(cols, chunk):
                input_data[feat] = col.number_input(feat, value=0.0, format="%.4f", step=0.001)

    st.markdown("")
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

    if predict_btn:
        try:
            row   = np.array([[input_data.get(f, 0.0) for f in feature_names]], dtype=np.float32)
            X     = scaler.transform(row)
            proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else 0.5
            pred  = int(proba >= opt_threshold)
            level = "Low" if proba < 0.35 else ("Moderate" if proba < 0.65 else "High")
            label = class_names[pred]

            st.markdown("---")
            st.subheader("📋 Prediction Results")
            r_col, g_col = st.columns([1, 1])

            with r_col:
                st.markdown(f"""
                <div class="metric-card"><b>Diagnosis</b><br>
                  <span style="font-size:1.5rem;font-weight:700">{label}</span></div>
                <div class="metric-card"><b>Risk Level</b><br>
                  <span class="risk-{level}">{level} Risk</span></div>
                <div class="metric-card"><b>Confidence</b><br>
                  <span style="font-size:1.3rem">{proba*100:.1f}%</span></div>
                <div class="metric-card"><b>Decision Threshold</b><br>
                  <span style="font-size:1rem">Using <b>{opt_threshold:.3f}</b>
                  {"(optimised for recall)" if opt_threshold != 0.5 else "(default)"}</span></div>
                """, unsafe_allow_html=True)

            with g_col:
                st.plotly_chart(make_gauge(proba, label), use_container_width=True)

            fig_bar = go.Figure(go.Bar(
                x=[f"P({class_names[0]})", f"P({class_names[1]})"],
                y=[round(1-proba, 4), round(proba, 4)],
                marker_color=["#2196F3", "#F44336"],
                text=[f"{(1-proba)*100:.1f}%", f"{proba*100:.1f}%"],
                textposition="outside",
            ))
            fig_bar.update_layout(yaxis_range=[0,1.15], height=260,
                                  title_text="Class Probability Distribution",
                                  margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(
                '<div class="disclaimer">⚠️ <b>Medical Disclaimer:</b> For research and '
                'educational purposes only. Not a substitute for professional medical advice, '
                'diagnosis, or treatment.</div>', unsafe_allow_html=True
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
