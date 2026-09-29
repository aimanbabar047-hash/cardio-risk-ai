# ============================================================
# CARDIORISK AI
# AI-Powered Cardiovascular Risk Assessment
# GIBC V2 - Applied Track
#
# Updated to match the notebook
# "Cardiovascular_Disease_data_analysis_updated.ipynb"
#
# The app reads everything the notebook saves in ./models :
#   cardiorisk_xgboost_pipeline.joblib  (deployed model)
#   model_metadata.json                 (threshold, input ranges, features)
#   model_metrics.json                  (CV, bootstrap, calibration, subgroups ...)
#   test_predictions.npz                (test-set probabilities for curves)
# so no metric or threshold is hardcoded here any more.
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp { background-color: #f7f9fc; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e4e7ec;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
        min-height: 125px;
    }

    .metric-title {
        font-size: 14px;
        color: #667085;
        font-weight: 600;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e4e7ec;
        margin-bottom: 20px;
    }

    .risk-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #e4e7ec;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }

    .risk-number {
        font-size: 58px;
        font-weight: 900;
        margin: 5px 0px;
    }

    .risk-label {
        font-size: 16px;
        color: #667085;
        font-weight: 600;
    }

    .info-box {
        background: #eef4ff;
        border-left: 5px solid #4f46e5;
        padding: 16px;
        border-radius: 10px;
        margin: 15px 0px;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 16px;
        border-radius: 10px;
        margin: 15px 0px;
    }

    .success-box {
        background: #ecfdf3;
        border-left: 5px solid #12b76a;
        padding: 16px;
        border-radius: 10px;
        margin: 15px 0px;
    }

    .small-text {
        color: #667085;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS (must mirror the notebook)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "cardiorisk_xgboost_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.npz"

# Feature groups used by the notebook's preprocessing pipeline
NUMERIC_FEATURES = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

FEATURE_LABELS = {
    "ap_hi": "Systolic Blood Pressure",
    "ap_lo": "Diastolic Blood Pressure",
    "age_years": "Age",
    "weight": "Weight",
    "height": "Height",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol Consumption",
    "active": "Physical Activity",
    "gender": "Gender (code 1 vs 2)",
}

LEVEL_LABELS = {1: "Normal", 2: "Above Normal", 3: "High"}

# Fallback values (identical to the notebook constants) used only if
# model_metadata.json is missing.
DEFAULT_META = {
    "model_name": "XGBoost",
    "features": [
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active",
    ],
    "threshold": 0.5,
    "target_recall": 0.80,
    "input_ranges": {
        "age_years": [30, 65],
        "height": [100, 250],
        "weight": [30, 200],
        "ap_hi": [70, 250],
        "ap_lo": [40, 150],
    },
    "training_age_range": [30.0, 65.0],
    "n_train": None,
    "n_test": None,
}


# ============================================================
# LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_predictions(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error("Unable to load the trained model.")
    st.caption(f"Expected file: {MODEL_PATH}")
    st.code(str(e))
    st.stop()

meta = {**DEFAULT_META, **(load_json(str(METADATA_PATH)) or {})}
metrics = load_json(str(METRICS_PATH)) or {}
preds = load_predictions(str(PREDICTIONS_PATH))

THRESHOLD = float(meta["threshold"])
TARGET_RECALL = float(meta.get("target_recall", 0.80))
FEATURES = list(meta["features"])
RANGES = {k: tuple(v) for k, v in meta["input_ranges"].items()}
MODEL_NAME = meta["model_name"]

if not METADATA_PATH.exists():
    st.sidebar.warning(
        "model_metadata.json not found - using default settings "
        "(threshold 0.50). Re-run the notebook (Step 18) to create it."
    )


# ============================================================
# HELPERS
# ============================================================

def pct(value, digits=2):
    return f"{value:.{digits}f}%"


def deployed_test_metrics(thr):
    """Test-set metrics of the deployed model at a given threshold (percent)."""
    if preds is None or "y_prob_deploy" not in preds:
        return None
    y = preds["y_test"]
    p = preds["y_prob_deploy"]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": (tp + tn) / len(y) * 100,
        "Precision": tp / (tp + fp) * 100 if (tp + fp) else float("nan"),
        "Recall": tp / (tp + fn) * 100 if (tp + fn) else float("nan"),
        "Specificity": tn / (tn + fp) * 100 if (tn + fp) else float("nan"),
        "ROC-AUC": roc_auc_score(y, p) * 100,
        "PR-AUC": average_precision_score(y, p) * 100,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def to_original(transformed_name):
    """'cat__cholesterol_2' -> 'cholesterol', 'num__ap_hi' -> 'ap_hi'."""
    base = transformed_name.split("__", 1)[-1]
    if base in NUMERIC_FEATURES:
        return base
    for f in CATEGORICAL_FEATURES:
        if base == f or base.startswith(f + "_"):
            return f
    return base


def get_pipelines(m):
    """Return the fitted sklearn pipelines inside the deployed model.

    The notebook may deploy either a plain pipeline or an isotonic
    CalibratedClassifierCV (which holds one fitted pipeline per CV fold).
    """
    if hasattr(m, "calibrated_classifiers_"):
        return [
            getattr(c, "estimator", None) or getattr(c, "base_estimator")
            for c in m.calibrated_classifiers_
        ]
    return [m]


def patient_contributions(m, frame):
    """SHAP-style contributions (log-odds) grouped by original feature."""
    import xgboost as xgb

    series = []
    for pipe in get_pipelines(m):
        pre = pipe.named_steps["preprocessor"]
        booster = pipe.named_steps["model"].get_booster()
        names = [to_original(n) for n in pre.get_feature_names_out()]
        contribs = booster.predict(
            xgb.DMatrix(pre.transform(frame)), pred_contribs=True
        )
        series.append(
            pd.Series(contribs[0, :-1], index=names).groupby(level=0).sum()
        )
    return pd.concat(series, axis=1).mean(axis=1)


def global_importance(m):
    """XGBoost importance grouped back to the original features."""
    series = []
    for pipe in get_pipelines(m):
        pre = pipe.named_steps["preprocessor"]
        mdl = pipe.named_steps["model"]
        names = [to_original(n) for n in pre.get_feature_names_out()]
        series.append(
            pd.Series(mdl.feature_importances_, index=names).groupby(level=0).sum()
        )
    return pd.concat(series, axis=1).mean(axis=1).sort_values()


def show_records(title, key, note=None, hide_index=True):
    """Display a table stored in model_metrics.json."""
    records = metrics.get(key)
    if not records:
        return
    st.subheader(title)
    st.dataframe(
        pd.DataFrame(records),
        use_container_width=True,
        hide_index=hide_index
    )
    if note:
        st.caption(note)


def metric_cards(cards):
    cols = st.columns(len(cards))
    for col, (title, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("<h2>❤️ CardioRisk AI</h2>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class="small-text">
        AI-powered cardiovascular risk research platform
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🩺 Risk Assessment",
            "📊 Model Analytics",
            "🧠 Explainability",
            "🔬 Methodology",
            "⚠️ Responsible AI"
        ]
    )

    st.divider()

    st.caption("GIBC V2 • Applied Track")
    st.caption("Machine Learning Research Prototype")


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<p class="main-title">❤️ CardioRisk AI</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="subtitle">
        AI-powered cardiovascular disease risk assessment using machine learning
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>What does CardioRisk AI do?</b><br><br>

        CardioRisk AI analyzes demographic, clinical and lifestyle-related
        features and uses a trained, calibrated machine learning pipeline to
        estimate cardiovascular disease risk. Predictions are flagged with a
        screening threshold chosen to catch most cases.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Platform Snapshot")

    n_train, n_test = meta.get("n_train"), meta.get("n_test")
    n_records = f"{n_train + n_test:,}" if n_train and n_test else "—"

    dm = deployed_test_metrics(THRESHOLD)
    auc_text = pct(dm["ROC-AUC"]) if dm else "—"

    metric_cards([
        ("Dataset Records (after cleaning)", n_records),
        ("Input Features", str(len(FEATURES))),
        ("Algorithms Evaluated", "3"),
        (f"{MODEL_NAME} ROC-AUC", auc_text),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Machine Learning Pipeline")

    pipeline_steps = [
        ("01", "Data Cleaning"),
        ("02", "Preprocessing"),
        ("03", "Train/Test"),
        ("04", "Tuning & CV"),
        ("05", "Calibration"),
        ("06", "Threshold"),
    ]

    for col, (number, label) in zip(st.columns(6), pipeline_steps):
        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center;">
                    <div class="metric-title">{number}</div>
                    <div style="font-size:17px;font-weight:700;margin-top:12px;">
                    {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Core Capabilities")

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            ### 🔍 Predictive ML

            Converts structured health-related features into a calibrated
            cardiovascular disease probability with a screening decision.
            """
        )

    with b:
        st.markdown(
            """
            ### 📊 Rigorous Evaluation

            Cross-validation, bootstrap confidence intervals, calibration
            and subgroup analysis on top of standard test metrics.
            """
        )

    with c:
        st.markdown(
            """
            ### 🧠 Explainable AI

            SHAP-style contributions show which inputs pushed an individual
            prediction up or down.
            """
        )


# ============================================================
# PAGE 2 — RISK ASSESSMENT
# ============================================================

elif page == "🩺 Risk Assessment":

    st.markdown(
        '<p class="main-title">🩺 Cardiovascular Risk Assessment</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="subtitle">
        Enter the required information to generate a model-based risk estimate.
        </p>
        """,
        unsafe_allow_html=True
    )

    age_lo, age_hi = (int(v) for v in RANGES["age_years"])
    h_lo, h_hi = (int(v) for v in RANGES["height"])
    w_lo, w_hi = (float(v) for v in RANGES["weight"])
    sbp_lo, sbp_hi = (int(v) for v in RANGES["ap_hi"])
    dbp_lo, dbp_hi = (int(v) for v in RANGES["ap_lo"])

    st.info(
        f"The model was trained on adults aged {age_lo}–{age_hi}. "
        "Inputs outside the ranges shown below are not accepted, because "
        "the model has no experience with them."
    )

    # ========================================================
    # QUICK DEMO PROFILES (fill the form with one click)
    # ========================================================

    def _clamp(v, lo, hi):
        return min(max(v, lo), hi)

    PROFILES = {
        "🟢 Low-risk example": dict(
            in_age=35, in_gender=1, in_height=170, in_weight=65.0,
            in_ap_hi=110, in_ap_lo=70, in_chol=1, in_gluc=1,
            in_smoke=0, in_alco=0, in_active=1),
        "🟡 Borderline example": dict(
            in_age=52, in_gender=2, in_height=172, in_weight=82.0,
            in_ap_hi=130, in_ap_lo=85, in_chol=2, in_gluc=1,
            in_smoke=0, in_alco=0, in_active=0),
        "🔴 High-risk example": dict(
            in_age=62, in_gender=1, in_height=160, in_weight=92.0,
            in_ap_hi=165, in_ap_lo=100, in_chol=3, in_gluc=3,
            in_smoke=1, in_alco=0, in_active=0),
    }

    DEFAULT_INPUTS = dict(
        in_age=_clamp(55, age_lo, age_hi), in_gender=1,
        in_height=_clamp(165, h_lo, h_hi), in_weight=_clamp(75.0, w_lo, w_hi),
        in_ap_hi=_clamp(140, sbp_lo, sbp_hi), in_ap_lo=_clamp(85, dbp_lo, dbp_hi),
        in_chol=1, in_gluc=1, in_smoke=0, in_alco=0, in_active=1)

    for _k, _v in DEFAULT_INPUTS.items():
        st.session_state.setdefault(_k, _v)

    def load_profile(name):
        for _k, _v in PROFILES[name].items():
            st.session_state[_k] = _v

    st.markdown("**Quick demo profiles** - fill the form with one click")
    pcols = st.columns(len(PROFILES))
    for _col, _name in zip(pcols, PROFILES):
        _col.button(_name, on_click=load_profile, args=(_name,),
                    use_container_width=True)

    # ========================================================
    # DEMOGRAPHICS
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <h3>👤 Demographics & Body Measurements</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age_years = st.number_input(
            "Age (years)",
            min_value=age_lo,
            max_value=age_hi,
            key="in_age",
            step=1
        )

    with col2:
        # The dataset does not document which code is female / male,
        # so the app shows the raw codes instead of guessing.
        gender = st.selectbox(
            "Gender code",
            [1, 2],
            key="in_gender",
            format_func=lambda x: f"Code {x}",
            help="The public dataset does not state which code is female "
                 "and which is male."
        )

    with col3:
        height = st.number_input(
            "Height (cm)",
            min_value=h_lo,
            max_value=h_hi,
            key="in_height",
            step=1
        )

    with col4:
        weight = st.number_input(
            "Weight (kg)",
            min_value=w_lo,
            max_value=w_hi,
            key="in_weight",
            step=0.1
        )

    bmi = float(weight / ((height / 100) ** 2))

    st.metric("Calculated BMI (informational, not a model input)", f"{bmi:.1f} kg/m²")

    # ========================================================
    # CLINICAL
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <h3>🩺 Clinical Measurements</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ap_hi = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=sbp_lo,
            max_value=sbp_hi,
            key="in_ap_hi",
            step=1
        )

    with col2:
        ap_lo = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=dbp_lo,
            max_value=dbp_hi,
            key="in_ap_lo",
            step=1
        )

    with col3:
        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            key="in_chol",
            format_func=lambda x: LEVEL_LABELS[x]
        )

    with col4:
        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            key="in_gluc",
            format_func=lambda x: LEVEL_LABELS[x]
        )

    bp_valid = ap_hi > ap_lo
    if not bp_valid:
        st.error("Systolic blood pressure must be higher than diastolic blood pressure.")

    # ========================================================
    # LIFESTYLE
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <h3>🏃 Lifestyle Factors</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        smoke = st.selectbox(
            "Smoking",
            [0, 1],
            key="in_smoke",
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col2:
        alco = st.selectbox(
            "Alcohol Consumption",
            [0, 1],
            key="in_alco",
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col3:
        active = st.selectbox(
            "Physical Activity",
            [0, 1],
            key="in_active",
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    st.divider()

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🔍 ANALYZE CARDIOVASCULAR RISK",
        type="primary",
        use_container_width=True,
        disabled=not bp_valid
    )

    if predict_button:

        # Only the columns the notebook trained on (age_group is NOT a
        # model input any more; it is derived and used for EDA only).
        values = {
            "age_years": float(age_years),
            "gender": int(gender),
            "height": int(height),
            "weight": float(weight),
            "ap_hi": int(ap_hi),
            "ap_lo": int(ap_lo),
            "cholesterol": int(cholesterol),
            "gluc": int(gluc),
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active),
        }
        input_data = pd.DataFrame([values])[FEATURES]

        try:

            risk_probability = float(model.predict_proba(input_data)[0, 1])
            risk_percentage = risk_probability * 100

            # Screening threshold from the notebook - NOT the default 0.5
            prediction = int(risk_probability >= THRESHOLD)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("AI Risk Assessment")

            result_col1, result_col2 = st.columns([1, 2])

            with result_col1:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="risk-label">MODEL-ESTIMATED PROBABILITY</div>
                        <div class="risk-number">{risk_percentage:.2f}%</div>
                        <div class="risk-label">Cardiovascular Disease</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col2:

                st.progress(float(min(max(risk_probability, 0.0), 1.0)))

                st.caption(
                    f"Screening threshold: {THRESHOLD * 100:.1f}% "
                    f"(chosen for about {TARGET_RECALL:.0%} recall)"
                )

                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="warning-box">
                        <h3>⚠️ Screening Result: Above Threshold (Higher Risk)</h3>
                        The estimated probability of <b>{risk_percentage:.2f}%</b>
                        is at or above the screening threshold of
                        <b>{THRESHOLD * 100:.1f}%</b>. This flag favours catching
                        possible cases, so false alarms are expected.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="success-box">
                        <h3>✓ Screening Result: Below Threshold (Lower Risk)</h3>
                        The estimated probability of <b>{risk_percentage:.2f}%</b>
                        is below the screening threshold of
                        <b>{THRESHOLD * 100:.1f}%</b>.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.caption(
                    "This is a machine-learning model estimate, "
                    "not a clinical diagnosis."
                )

            # ------------------------------------------------
            # WHY THIS PREDICTION? (SHAP-style contributions)
            # ------------------------------------------------

            try:
                contrib = patient_contributions(model, input_data)
                contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Why this prediction?")

                chart_df = (
                    contrib.rename(index=FEATURE_LABELS)
                    .sort_values()
                    .to_frame("Contribution (log-odds)")
                )
                st.bar_chart(chart_df)

                top = contrib.head(3)
                drivers = ", ".join(
                    f"{FEATURE_LABELS.get(k, k)} "
                    f"({'raises' if v > 0 else 'lowers'} risk)"
                    for k, v in top.items()
                )
                st.caption(
                    f"Strongest drivers for this input: {drivers}. "
                    "Values are log-odds of the underlying XGBoost model; "
                    "positive values push towards cardiovascular disease. "
                    "They explain model behaviour, not biological causes."
                )
            except Exception:
                st.caption("Per-patient explanation is not available for this model file.")

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Patient Input Summary")

            summary = pd.DataFrame({
                "Feature": [
                    "Age", "Gender code", "Height", "Weight", "BMI (not a model input)",
                    "Systolic BP", "Diastolic BP", "Cholesterol", "Glucose",
                    "Smoking", "Alcohol", "Physical Activity"
                ],
                "Value": [
                    f"{int(age_years)} years",
                    f"Code {int(gender)}",
                    f"{int(height)} cm",
                    f"{float(weight):.1f} kg",
                    f"{bmi:.1f} kg/m²",
                    f"{int(ap_hi)} mmHg",
                    f"{int(ap_lo)} mmHg",
                    LEVEL_LABELS[int(cholesterol)],
                    LEVEL_LABELS[int(gluc)],
                    "Yes" if smoke else "No",
                    "Yes" if alco else "No",
                    "Yes" if active else "No",
                ]
            })

            st.dataframe(summary, use_container_width=True, hide_index=True)

            # ------------------------------------------------
            # MODEL INFORMATION (computed from saved test predictions)
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Model Information")

            dm = deployed_test_metrics(THRESHOLD)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model", MODEL_NAME)
            m2.metric("ROC-AUC (test)", pct(dm["ROC-AUC"]) if dm else "—")
            m3.metric("PR-AUC (test)", pct(dm["PR-AUC"]) if dm else "—")
            m4.metric("Recall @ threshold", pct(dm["Recall"]) if dm else "—")

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))


# ============================================================
# PAGE 3 — MODEL ANALYTICS
# ============================================================

elif page == "📊 Model Analytics":

    st.markdown(
        '<p class="main-title">📊 Model Analytics</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="subtitle">
        Test-set performance, robustness checks, calibration and subgroup analysis
        </p>
        """,
        unsafe_allow_html=True
    )

    if not metrics and preds is None:
        st.warning(
            "No analytics files found in the models folder. Run the notebook "
            "through Step 18 to create model_metrics.json and test_predictions.npz."
        )

    tab_perf, tab_curves, tab_robust, tab_calib, tab_group = st.tabs([
        "Test Performance",
        "Curves & Threshold",
        "Robustness (CV & CI)",
        "Calibration",
        "Subgroups",
    ])

    # --------------------------------------------------------
    # TAB 1 - TEST PERFORMANCE
    # --------------------------------------------------------

    with tab_perf:

        dm = deployed_test_metrics(THRESHOLD)

        if dm:
            st.subheader(f"Deployed model: {MODEL_NAME}")
            st.caption(f"Held-out test set, screening threshold {THRESHOLD:.3f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", pct(dm["Accuracy"]))
            c2.metric("ROC-AUC", pct(dm["ROC-AUC"]))
            c3.metric("PR-AUC", pct(dm["PR-AUC"]))

            c4, c5, c6 = st.columns(3)
            c4.metric("Precision", pct(dm["Precision"]))
            c5.metric("Recall", pct(dm["Recall"]))
            c6.metric("Specificity", pct(dm["Specificity"]))

        records = metrics.get("test_metrics_percent")
        if records:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Model Comparison (threshold 0.50)")

            comparison_df = pd.DataFrame(records).set_index("Model")
            st.dataframe(comparison_df.round(2), use_container_width=True)

            selected_metric = st.selectbox(
                "Select evaluation metric",
                [c for c in comparison_df.columns]
            )
            st.bar_chart(comparison_df[selected_metric].sort_values())

        st.info(
            "Evaluation was performed on the held-out test set. The comparison "
            "table uses the default 0.50 threshold for every model; the deployed "
            "model additionally uses the screening threshold shown above."
        )

    # --------------------------------------------------------
    # TAB 2 - CURVES & THRESHOLD
    # --------------------------------------------------------

    with tab_curves:

        if preds is None:
            st.warning("test_predictions.npz not found - curves are unavailable.")
        else:
            y_test = preds["y_test"]

            prob_keys = [
                ("Logistic Regression", "y_prob_lr"),
                ("Random Forest", "y_prob_rf"),
                ("XGBoost (default)", "y_prob_xgb"),
                ("XGBoost (tuned)", "y_prob_xgb_tuned"),
                (f"Deployed: {MODEL_NAME}", "y_prob_deploy"),
            ]
            prob_keys = [(n, k) for n, k in prob_keys if k in preds]

            grid = np.linspace(0, 1, 201)

            roc_df = pd.DataFrame(index=pd.Index(grid, name="False positive rate"))
            pr_df = pd.DataFrame(index=pd.Index(grid, name="Recall"))

            for name, key in prob_keys:
                fpr, tpr, _ = roc_curve(y_test, preds[key])
                roc_df[f"{name} (AUC {roc_auc_score(y_test, preds[key]):.3f})"] = np.interp(grid, fpr, tpr)

                prec, rec, _ = precision_recall_curve(y_test, preds[key])
                pr_df[name] = np.interp(grid, rec[::-1], prec[::-1])

            roc_df["Random classifier"] = grid

            left, right = st.columns(2)
            with left:
                st.subheader("ROC Curves")
                st.line_chart(roc_df)
                st.caption("Y axis: true positive rate")
            with right:
                st.subheader("Precision-Recall Curves")
                st.line_chart(pr_df)
                st.caption("Y axis: precision")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Screening Threshold Explorer")

            st.markdown(
                f"""
                <div class="info-box">
                A fixed 0.50 threshold misses many patients. The deployed
                threshold (<b>{THRESHOLD:.3f}</b>) was chosen from
                out-of-fold training predictions for about
                <b>{TARGET_RECALL:.0%}</b> recall. Higher recall always brings
                more false alarms - move the slider to see the trade-off.
                </div>
                """,
                unsafe_allow_html=True
            )

            thr = st.slider(
                "Decision threshold",
                min_value=0.05,
                max_value=0.95,
                value=float(np.clip(THRESHOLD, 0.05, 0.95)),
                step=0.01
            )

            dm_thr = deployed_test_metrics(thr)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recall", pct(dm_thr["Recall"]))
            c2.metric("Precision", pct(dm_thr["Precision"]))
            c3.metric("Specificity", pct(dm_thr["Specificity"]))
            c4.metric("Accuracy", pct(dm_thr["Accuracy"]))

            cm_df = pd.DataFrame(
                [[dm_thr["tn"], dm_thr["fp"]], [dm_thr["fn"], dm_thr["tp"]]],
                index=["Actual: No CVD", "Actual: CVD"],
                columns=["Predicted: No CVD", "Predicted: CVD"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(
                f"Missed patients (false negatives): {dm_thr['fn']:,} • "
                f"False alarms (false positives): {dm_thr['fp']:,}"
            )

        show_records(
            "Threshold Trade-off Table",
            "threshold_tradeoff",
            "Thresholds were derived from training-set out-of-fold predictions; "
            "results are on the test set."
        )

    # --------------------------------------------------------
    # TAB 3 - ROBUSTNESS
    # --------------------------------------------------------

    with tab_robust:

        cv_records = metrics.get("cross_validation_percent")
        if cv_records:
            st.subheader("5-Fold Cross-Validation (mean ± std, %)")
            cv_df = pd.DataFrame(cv_records)
            cv_df["Value"] = cv_df.apply(
                lambda r: f"{r['Mean']:.2f} ± {r['Std']:.2f}", axis=1
            )
            cv_table = cv_df.pivot(index="Model", columns="Metric", values="Value")
            order = [c for c in
                     ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
                     if c in cv_table.columns]
            st.dataframe(cv_table[order], use_container_width=True)

        show_records(
            "ROC-AUC with 95% Bootstrap Confidence Interval",
            "roc_auc_bootstrap_ci",
            "Overlapping intervals mean the ranking between models is not certain."
        )

        show_records(
            "Paired ROC-AUC Differences",
            "paired_auc_differences",
            "If the interval includes 0, the difference between the two models "
            "cannot be distinguished from noise."
        )

        if not metrics:
            st.warning("model_metrics.json not found.")

    # --------------------------------------------------------
    # TAB 4 - CALIBRATION
    # --------------------------------------------------------

    with tab_calib:

        st.markdown(
            """
            <div class="info-box">
            The app shows a probability, so calibration matters: "80%" should
            mean that about 80 of 100 similar patients have the disease.
            Brier score and expected calibration error (ECE) - lower is
            better - are compared with and without isotonic calibration.
            </div>
            """,
            unsafe_allow_html=True
        )

        show_records("Calibration Metrics", "calibration")

        if preds is not None and "y_prob_deploy" in preds:
            from sklearn.calibration import calibration_curve

            frac_pos, mean_pred = calibration_curve(
                preds["y_test"], preds["y_prob_deploy"],
                n_bins=10, strategy="quantile"
            )
            calib_df = pd.DataFrame(
                {
                    "Observed fraction with CVD": frac_pos,
                    "Perfectly calibrated": mean_pred,
                },
                index=pd.Index(np.round(mean_pred, 3), name="Mean predicted probability"),
            )
            st.subheader("Reliability Diagram (deployed model)")
            st.line_chart(calib_df)

    # --------------------------------------------------------
    # TAB 5 - SUBGROUPS
    # --------------------------------------------------------

    with tab_group:

        show_records(
            "Subgroup Performance",
            "subgroups",
            "Small groups have wide uncertainty - check the group size (n). "
            "Age groups are derived from age and are not model inputs."
        )

        subgroup_records = metrics.get("subgroups")
        if subgroup_records:
            sg = pd.DataFrame(subgroup_records)
            metric_choice = st.selectbox("Metric", ["ROC-AUC", "Recall", "Precision", "Specificity"])
            if metric_choice in sg.columns:
                st.bar_chart(sg.set_index("Group")[metric_choice])
        else:
            st.warning("Subgroup results not found in model_metrics.json.")


# ============================================================
# PAGE 4 — EXPLAINABILITY
# ============================================================

elif page == "🧠 Explainability":

    st.markdown(
        '<p class="main-title">🧠 Model Explainability</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="subtitle">
        Understanding which inputs drive the XGBoost model
        </p>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Global Feature Importance")

    try:
        importance = global_importance(model)
        plot_data = importance.rename(index=FEATURE_LABELS).to_frame("Importance")

        st.bar_chart(plot_data)

        display_df = (
            importance.sort_values(ascending=False)
            .rename(index=FEATURE_LABELS)
            .mul(100)
            .round(2)
            .rename("Importance (%)")
            .reset_index()
            .rename(columns={"index": "Feature"})
        )
        display_df.columns = ["Feature", "Importance (%)"]

        st.subheader("Feature Importance Table")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.caption(
            "One-hot columns are grouped back into their original feature, so "
            "'Smoker' and 'Non-Smoker' are no longer shown as separate rows. "
            "Values come directly from the deployed model file."
        )

    except Exception as e:
        st.warning("Feature importance could not be read from the model file.")
        st.code(str(e))

    st.markdown(
        """
        <div class="info-box">

        <b>Explain a single patient</b><br><br>

        The <b>Risk Assessment</b> page shows a "Why this prediction?" chart
        for every input, based on XGBoost's built-in SHAP-style contributions
        (log-odds). Positive values push the prediction towards cardiovascular
        disease; negative values push it away.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        <b>How should feature importance be interpreted?</b><br><br>

        Feature importance describes how strongly variables contributed
        to the trained model's predictive behavior.

        It does <b>not</b> establish biological causation and does not
        mean that an individual feature independently determines
        cardiovascular disease.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 5 — METHODOLOGY
# ============================================================

elif page == "🔬 Methodology":

    st.markdown(
        '<p class="main-title">🔬 Methodology</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="subtitle">
        End-to-end machine learning workflow
        </p>
        """,
        unsafe_allow_html=True
    )

    age_lo, age_hi = (int(v) for v in RANGES["age_years"])
    n_train, n_test = meta.get("n_train"), meta.get("n_test")
    records_text = (
        f"{n_train + n_test:,} records remained after cleaning"
        if n_train and n_test else "Records were cleaned as described below"
    )

    steps = [
        (
            "01", "Dataset Preparation",
            f"Public cardiovascular disease dataset (70,000 raw records). "
            f"{records_text}."
        ),
        (
            "02", "Data Cleaning",
            "Records with implausible blood pressure (systolic outside 70–250, "
            "diastolic outside 40–150, or systolic not above diastolic) and "
            "clearly invalid height (100–250 cm) or weight (30–200 kg) were removed."
        ),
        (
            "03", "Feature Engineering",
            f"Age in days was converted to years. The app accepts ages "
            f"{age_lo}–{age_hi}, the range covered by the training data. Age "
            f"groups (<40, 40-49, 50-59, 60-64) are derived from age and used "
            f"only for exploration and subgroup analysis - they are not model inputs."
        ),
        (
            "04", "Train/Test Split",
            "80/20 stratified split with a fixed random seed. The test set was "
            "not used for tuning or threshold selection."
        ),
        (
            "05", "Preprocessing",
            "Numeric features are standardised; categorical features are "
            "one-hot encoded (binary features keep a single column) inside one "
            "scikit-learn pipeline."
        ),
        (
            "06", "Model Development & Tuning",
            "Logistic Regression, Random Forest and XGBoost were compared. "
            "XGBoost was tuned with a randomized search (3-fold CV, ROC-AUC) on "
            "the training set only, and the tuned variant was used only if it "
            "beat the default on the same folds."
        ),
        (
            "07", "Evaluation",
            "Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC on the test "
            "set, plus 5-fold cross-validation and bootstrap confidence "
            "intervals to check whether differences between models are real."
        ),
        (
            "08", "Calibration",
            "Isotonic calibration was fitted on the training set and deployed "
            "only if it lowered the Brier score."
        ),
        (
            "09", "Screening Threshold",
            f"The decision threshold ({THRESHOLD:.3f}) was chosen from "
            f"out-of-fold training predictions for about {TARGET_RECALL:.0%} "
            f"recall, accepting more false alarms to miss fewer patients."
        ),
        (
            "10", "Explainability & Subgroups",
            "SHAP-style contributions (grouped by original feature) and "
            "performance by gender code and age group."
        ),
        (
            "11", "Deployment",
            "The trained pipeline, its metadata and metrics are saved by the "
            "notebook and read by this Streamlit application."
        ),
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>{number} — {title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Machine Learning Architecture")

    st.code(
        """
Raw Dataset (70,000)
     │
     ▼
Data Cleaning (BP, height, weight)
     │
     ▼
Feature Engineering (age in years)
     │
     ▼
Stratified Train / Test Split
     │
     ▼
Preprocessing Pipeline
     │
     ├── Logistic Regression
     ├── Random Forest
     └── XGBoost ── Randomized Search (CV)
             │
             ▼
   5-Fold CV • Bootstrap CI
             │
             ▼
   Isotonic Calibration (if better)
             │
             ▼
   Screening Threshold (target recall)
             │
             ▼
   SHAP • Subgroup Analysis
             │
             ▼
       CardioRisk AI
       Web Application
        """,
        language="text"
    )


# ============================================================
# PAGE 6 — RESPONSIBLE AI
# ============================================================

elif page == "⚠️ Responsible AI":

    st.markdown(
        '<p class="main-title">⚠️ Responsible AI</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="subtitle">
        Scope, limitations and responsible use
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">

        <h3>Research Prototype — Not a Medical Diagnosis</h3>

        CardioRisk AI is an educational and research prototype.
        Its predictions should not be used as a substitute for
        professional medical evaluation, diagnosis or treatment.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Important Limitations")

    age_lo, age_hi = (int(v) for v in RANGES["age_years"])

    limitations = [
        "The model was developed using a public dataset rather than prospectively collected clinical data.",
        "The model has not undergone clinical validation.",
        f"The training data only covers ages {age_lo}–{age_hi}; the app rejects inputs outside the ranges seen in training.",
        f"The screening threshold ({THRESHOLD * 100:.1f}%) favours recall, so many flagged people will not have the disease (false alarms).",
        "Differences between the compared models are small; see the confidence intervals on the Model Analytics page.",
        "Performance can differ between subgroups (gender code, age group); check the Subgroups tab.",
        "The dataset does not document which gender code is female or male, so the app shows the raw codes.",
        "Model probability estimates should not be interpreted as individual clinical risk scores.",
        "Feature importance and SHAP-style contributions describe model behavior and do not prove biological causation.",
        "The application is intended for research, education and demonstration purposes.",
    ]

    for item in limitations:
        st.markdown(f"• {item}")

    st.subheader("Responsible Use")

    st.markdown(
        """
        CardioRisk AI demonstrates how machine learning can transform
        structured health-related data into an interpretable predictive
        workflow.

        Human clinical expertise remains essential for any real-world
        healthcare decision.
        """
    )

    st.divider()

    st.caption(
        "CardioRisk AI • Machine Learning Research Prototype • GIBC V2"
    )
