"""
Type-2 Diabetes Prediction App (Streamlit)
Uses 6 ML models from 5-Fold Cross-Validation (Fold 1)
Run: streamlit run app.py
"""

import os
import json
import warnings
import logging

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

# ---------------------------------------------------------------------------
# Suppress noisy warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)
os.environ["PYTORCH_LIGHTNING_SUPPRESS_MODEL_SUMMARY"] = "1"
os.environ["PL_DISABLE_RICH_LOGGING"] = "1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_NAMES = [
    "No. of Pregnancy", "Age", "BMI", "BP(Systolic)", "BP(Diastolic)",
    "DiabetesPedigreeFunction", "Insulin", "Skin Thickness(mm)", "Glucose",
]

MODEL_DISPLAY_NAMES = {
    "KNN":                 "K-Nearest Neighbors (KNN)",
    "SVM":                 "Support Vector Machine (SVM)",
    "XGBoost":             "XGBoost",
    "LightGBM":            "LightGBM",
    "MLP_PyTorch":         "MLP Neural Network (PyTorch)",
    "Tabular_Transformer": "Tabular Transformer (FTTransformer)",
}


# ---------------------------------------------------------------------------
# MLP Architecture (must match training notebook exactly)
# ---------------------------------------------------------------------------
class DiabetesMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=(128, 64, 32), dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev_dim = h
        layers += [nn.Linear(prev_dim, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Model Loading (cached so models load only once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models...")
def load_all_models():
    """Load all 6 ML models once and cache them in memory."""
    models = {}

    # --- KNN & SVM: trained on imputed + scaled data ---
    for key in ["KNN", "SVM"]:
        try:
            folder = os.path.join(MODELS_DIR, key)
            models[key] = {
                "model":   joblib.load(os.path.join(folder, "best_model.pkl")),
                "imputer": joblib.load(os.path.join(folder, "imputer.pkl")),
                "scaler":  joblib.load(os.path.join(folder, "scaler.pkl")),
                "type":    "sklearn",
            }
        except Exception as e:
            st.warning(f"Could not load {key}: {e}")

    # --- XGBoost & LightGBM: trained on imputed data only (NO scaling) ---
    for key in ["XGBoost", "LightGBM"]:
        try:
            folder = os.path.join(MODELS_DIR, key)
            models[key] = {
                "model":   joblib.load(os.path.join(folder, "best_model.pkl")),
                "imputer": joblib.load(os.path.join(folder, "imputer.pkl")),
                "scaler":  joblib.load(os.path.join(folder, "scaler.pkl")),
                "type":    "tree",  # tree-based: skip scaler during prediction
            }
        except Exception as e:
            st.warning(f"Could not load {key}: {e}")

    # --- PyTorch MLP ---
    try:
        folder = os.path.join(MODELS_DIR, "MLP_PyTorch")
        with open(os.path.join(folder, "hyperparameters.json")) as f:
            hp = json.load(f)
        hidden = tuple(int(x) for x in hp["hidden_layers"].strip("()").split(",") if x.strip())
        dropout = float(hp["dropout_rate"])
        device = torch.device("cpu")
        mlp = DiabetesMLP(input_dim=len(FEATURE_NAMES), hidden_layers=hidden, dropout_rate=dropout)
        ckpt = torch.load(os.path.join(folder, "best_model.pth"), map_location=device, weights_only=False)
        mlp.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        mlp.eval()
        models["MLP_PyTorch"] = {
            "model": mlp, "imputer": joblib.load(os.path.join(folder, "imputer.pkl")),
            "scaler": joblib.load(os.path.join(folder, "scaler.pkl")),
            "type": "mlp", "device": device,
        }
    except Exception as e:
        st.warning(f"Could not load MLP_PyTorch: {e}")

    # --- Tabular Transformer (FTTransformer) ---
    try:
        from pytorch_tabular import TabularModel
        folder = os.path.join(MODELS_DIR, "Tabular_Transformer")
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": "cpu"})
        try:
            tab_model = TabularModel.load_model(os.path.join(folder, "model"))
        finally:
            torch.load = _orig
        models["Tabular_Transformer"] = {
            "model": tab_model, "imputer": joblib.load(os.path.join(folder, "imputer.pkl")),
            "scaler": joblib.load(os.path.join(folder, "scaler.pkl")),
            "type": "tabular_transformer",
        }
    except Exception as e:
        st.warning(f"Could not load Tabular Transformer: {e}")

    return models


# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------
def predict(model_key: str, features: dict, models: dict) -> dict:
    """Run prediction with a given model and return result dict."""
    entry = models[model_key]
    model, imputer, scaler = entry["model"], entry["imputer"], entry["scaler"]

    # Build DataFrame
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    
    # Safe Imputation (bypasses sklearn version mismatch errors)
    df_imp = df.copy()
    if hasattr(imputer, "statistics_"):
        for i, col in enumerate(FEATURE_NAMES):
            df_imp[col] = df_imp[col].fillna(imputer.statistics_[i])
            
    # Scale
    df_sc = pd.DataFrame(scaler.transform(df_imp), columns=FEATURE_NAMES)

    if entry["type"] == "tree":
        # XGBoost & LightGBM were trained on imputed data only (no scaling).
        proba = model.predict_proba(df_imp.values)[:, 1][0]

    elif entry["type"] == "sklearn":
        # KNN & SVM were trained on imputed + scaled data.
        proba = model.predict_proba(df_sc.values)[:, 1][0]

    elif entry["type"] == "mlp":
        tensor = torch.FloatTensor(df_sc.values).to(entry["device"])
        with torch.no_grad():
            proba = model(tensor).cpu().numpy().flatten()[0]

    elif entry["type"] == "tabular_transformer":
        df_pred = df_sc.copy()
        df_pred["Type-2 Diabetic"] = 0  # dummy target column
        out = model.predict(df_pred)
        prob_col = [c for c in out.columns if c.endswith("_1_probability")]
        pred_col = [c for c in out.columns if c.endswith("_prediction")]
        if prob_col:
            proba = float(out[prob_col[0]].iloc[0])
        elif pred_col:
            proba = float(out[pred_col[0]].iloc[0])
        else:
            proba = 0.5

    proba = float(np.clip(proba, 0.0, 1.0))
    return {
        "model_name":  MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "prediction":  "Diabetic" if proba >= 0.5 else "Non-Diabetic",
        "is_diabetic": proba >= 0.5,
        "probability": round(proba * 100, 2),
    }


# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Type-2 Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }
    .block-container { max-width: 1100px; padding-top: 2rem; }

    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e40af, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .app-header p {
        color: #64748b;
        font-size: 0.95rem;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #eff6ff, #ede9fe);
        color: #4338ca;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.3rem 0.9rem;
        border-radius: 99px;
        border: 1px solid #c7d2fe;
        margin-bottom: 0.8rem;
    }

    /* Section headers */
    .section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.6rem;
    }

    /* Input cards */
    .input-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* BMI display */
    .bmi-display {
        background: linear-gradient(135deg, #eff6ff, #f0fdf4);
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .bmi-display .value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1e40af;
    }
    .bmi-display .label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-diabetic {
        background: linear-gradient(135deg, #fef2f2, #fff1f2);
        border: 2px solid #fca5a5;
    }
    .result-healthy {
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border: 2px solid #86efac;
    }
    .result-verdict {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .result-verdict.diabetic { color: #dc2626; }
    .result-verdict.healthy { color: #16a34a; }
    .result-prob {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-prob.diabetic { color: #ef4444; }
    .result-prob.healthy { color: #22c55e; }
    .result-model {
        font-size: 0.75rem;
        color: #64748b;
        background: white;
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        margin-top: 0.5rem;
    }

    /* Probability bar */
    .prob-bar-container {
        background: #e2e8f0;
        border-radius: 99px;
        height: 12px;
        overflow: hidden;
        margin-top: 1rem;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 1s ease;
    }
    .prob-bar-fill.diabetic {
        background: linear-gradient(90deg, #fbbf24, #ef4444);
    }
    .prob-bar-fill.healthy {
        background: linear-gradient(90deg, #34d399, #22c55e);
    }
    .prob-bar-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.65rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #f1f5f9;
        margin-top: 2rem;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------------------
loaded_models = load_all_models()

# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="badge">🧬 AI-Powered Clinical Prediction</div>
    <h1>Type-2 Diabetes Predictor</h1>
    <p>Select a model, enter patient clinical data, and get an instant risk assessment.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UI — Model Selector
# ---------------------------------------------------------------------------
st.markdown('<p class="section-label">🤖 Select Prediction Model</p>', unsafe_allow_html=True)

model_keys = list(loaded_models.keys())
model_labels = [MODEL_DISPLAY_NAMES.get(k, k) for k in model_keys]
selected_label = st.selectbox("Model", model_labels, label_visibility="collapsed")
selected_key = model_keys[model_labels.index(selected_label)]

st.markdown("")  # spacer

# ---------------------------------------------------------------------------
# UI — Patient Features
# ---------------------------------------------------------------------------
st.markdown('<p class="section-label">📋 Patient Clinical Data</p>', unsafe_allow_html=True)

# Row 1: Demographics + Body metrics
r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns([1, 1, 1, 0.8, 0.8, 1])

with r1c1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, step=1,
                                  help="Number of times pregnant (0–20)")
with r1c2:
    age = st.number_input("Age (yrs)", min_value=18, max_value=120, value=45, step=1,
                          help="Patient's age in years (18–120)")
with r1c3:
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5,
                             help="Patient's weight in kilograms")
with r1c4:
    feet = st.number_input("Height (ft)", min_value=3, max_value=8, value=5, step=1,
                           help="Feet portion of height")
with r1c5:
    inches = st.number_input("Height (in)", min_value=0.0, max_value=11.9, value=7.0, step=0.5,
                             help="Inches portion of height")

# Auto-calculate BMI
height_m = ((feet * 12) + inches) * 0.0254
bmi = round(weight / (height_m ** 2), 2) if height_m > 0 else 0.0

with r1c6:
    st.markdown(f"""
    <div class="bmi-display">
        <div class="label">Calculated BMI</div>
        <div class="value">{bmi}</div>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Clinical features
r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns(6)

with r2c1:
    bp_sys = st.number_input("BP Systolic", min_value=70, max_value=250, value=120, step=1,
                             help="Upper BP reading in mmHg (70–250)")
with r2c2:
    bp_dia = st.number_input("BP Diastolic", min_value=20, max_value=150, value=80, step=1,
                             help="Lower BP reading in mmHg (20–150)")
with r2c3:
    glucose = st.number_input("Glucose (mg/dL)", min_value=20, max_value=500, value=140, step=1,
                              help="Plasma glucose concentration (20–500)")
with r2c4:
    insulin = st.number_input("Insulin (µU/mL)", min_value=0, max_value=900, value=20, step=1,
                              help="2-Hour serum insulin level (0–900)")
with r2c5:
    skin = st.number_input("Skin Thick. (mm)", min_value=0.0, max_value=500.0, value=300.0, step=0.01,
                           help="Triceps skin fold thickness (0–500)")
with r2c6:
    dpf = st.number_input("Pedigree Func.", min_value=0.0, max_value=10.0, value=1.5, step=0.01,
                          help="Diabetes Pedigree Function — genetic risk score (0–10)")

st.markdown("")  # spacer

# ---------------------------------------------------------------------------
# UI — Predict Button
# ---------------------------------------------------------------------------
predict_clicked = st.button("⚡  Run Prediction", use_container_width=True, type="primary")

if predict_clicked:
    features = {
        "No. of Pregnancy": pregnancies,
        "Age": age,
        "BMI": bmi,
        "BP(Systolic)": bp_sys,
        "BP(Diastolic)": bp_dia,
        "DiabetesPedigreeFunction": dpf,
        "Insulin": insulin,
        "Skin Thickness(mm)": skin,
        "Glucose": glucose,
    }

    with st.spinner("Running prediction..."):
        result = predict(selected_key, features, loaded_models)

    # --- Display Result ---
    is_d = result["is_diabetic"]
    cls = "diabetic" if is_d else "healthy"
    icon = "🔴" if is_d else "🟢"
    card_cls = "result-diabetic" if is_d else "result-healthy"

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div style="font-size: 2.5rem;">{icon}</div>
        <div class="result-verdict {cls}">{result['prediction']}</div>
        <div class="result-prob {cls}">{result['probability']}%</div>
        <div style="font-size: 0.8rem; color: #64748b;">Probability of Type-2 Diabetes</div>
        <div class="result-model">Model: {result['model_name']}</div>
        <div class="prob-bar-container">
            <div class="prob-bar-fill {cls}" style="width: {result['probability']}%;"></div>
        </div>
        <div class="prob-bar-labels">
            <span>Non-Diabetic</span>
            <span>Diabetic</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    Type-2 Diabetes Prediction · 5-Fold Cross-Validation (Fold 1)
</div>
""", unsafe_allow_html=True)
