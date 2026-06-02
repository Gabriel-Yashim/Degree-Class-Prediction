"""
============================================================
OULAD Degree Class Prediction System  —  Streamlit App
============================================================
Prerequisites (run retrain_top20.py first):
  - stacking_top20_model.pkl
  - top20_feature_names.json
  - top20_feature_importances.json
  - label_map.json
============================================================
"""

import streamlit as st
import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, json
from pathlib import Path


st.set_page_config(
    page_title="Degree Class Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
  .stApp { background-color:#0f1117; color:#e0e0e0; }

  section[data-testid="stSidebar"] {
    background-color:#1a1d27;
    border-right:1px solid #2a2d3a;
  }

  .result-card {
    border-radius:14px; padding:28px 20px;
    margin:10px 0; text-align:center;
    font-family:'Segoe UI',sans-serif;
  }
  .card-distinction{ background:linear-gradient(135deg,#0d2a47,#1a4a8a);
                     border:2px solid #4e9af1; }
  .card-pass       { background:linear-gradient(135deg,#0d2e0d,#1a4d1a);
                     border:2px solid #4ef18a; }
  .card-fail       { background:linear-gradient(135deg,#2e0d0d,#4d1a1a);
                     border:2px solid #f14e4e; }
  .card-withdrawn  { background:linear-gradient(135deg,#2e280d,#4d421a);
                     border:2px solid #f1c94e; }

  .res-emoji  { font-size:3rem; }
  .res-label  { font-size:2.4rem; font-weight:800; margin:6px 0 2px; }
  .res-sub    { font-size:.95rem; color:#999; }
  .res-conf   { font-size:1.7rem; font-weight:700; margin-top:12px; }

  .box-info  { background:#1a1d27; border-left:4px solid #4e9af1;
               border-radius:6px; padding:12px 16px; margin:6px 0; font-size:.9rem; }
  .box-warn  { background:#1a1d27; border-left:4px solid #f1c94e;
               border-radius:6px; padding:12px 16px; margin:6px 0; font-size:.9rem; }
  .box-risk  { background:#1a1d27; border-left:4px solid #f14e4e;
               border-radius:6px; padding:12px 16px; margin:6px 0; font-size:.9rem; }
  .box-ok    { background:#1a1d27; border-left:4px solid #4ef18a;
               border-radius:6px; padding:12px 16px; margin:6px 0; font-size:.9rem; }

  .sec-hdr {
    font-size:1.05rem; font-weight:700; color:#4e9af1;
    border-bottom:1px solid #2a2d3a;
    padding-bottom:5px; margin:18px 0 10px;
  }

  div[data-testid="metric-container"] {
    background:#1a1d27; border:1px solid #2a2d3a;
    border-radius:8px; padding:10px;
  }

  #MainMenu{visibility:hidden;} footer{visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# LOAD ARTIFACTS
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = joblib.load("stacking_top20_degree_class_model.pkl")
    top20 = json.load(open("top20_feature_names_degree_class.json"))
    imp_data = json.load(open("top20_feature_importances_degree_class.json"))
    label_map = json.load(open("label_map_degree_class.json"))
    importances = [d["importance"] for d in imp_data]
    return model, top20, importances, label_map


try:
    model, TOP20, TOP20_IMP, LABEL_MAP = load_artifacts()
    LOADED = True
except FileNotFoundError as e:
    LOADED = False
    ERR = str(e)


# CONSTANTS
CLASS_ORDER = ["Pass", "Third Class", "Second Class Lower", "Second Class Upper", "First Class"]

CLASS_COLOR = {
    "First Class": "#4e9af1",
    "Second Class Upper": "#4ef18a",
    "Second Class Lower": "#f1c94e",
    "Third Class": "#ff8c42",
    "Pass": "#f14e4e",
}

CLASS_CARD = {
    "First Class": "card-first",
    "Second Class Upper": "card-second-upper",
    "Second Class Lower": "card-second-lower",
    "Third Class": "card-third",
    "Pass": "card-pass",
}

CLASS_EMOJI = {
    "First Class": "🥇",
    "Second Class Upper": "🥈",
    "Second Class Lower": "🥉",
    "Third Class": "📜",
    "Pass": "✅",
}

DEGREE_STYLES = {
    "First Class": {"css": "first-class", "emoji": "🥇", "color": "#4e9af1"},
    "Second Class Upper": {"css": "second-upper", "emoji": "🥈", "color": "#4ef18a"},
    "Second Class Lower": {"css": "second-lower", "emoji": "🥉", "color": "#f1c94e"},
    "Third Class": {"css": "third-class", "emoji": "📜", "color": "#ff8c42"},
    "Pass": {"css": "pass-class", "emoji": "✅", "color": "#f14e4e"},
}

FEAT_META = {
    "clicks_late": (
        "Late-Phase Clicks (days 151+)",
        "VLE clicks in the final third of the module",
        0,
        10000,
        500,
    ),
    "clicks_mid": (
        "Mid-Phase Clicks (days 76–150)",
        "VLE clicks in the middle phase of the module",
        0,
        10000,
        400,
    ),
    "clicks_early": (
        "Early-Phase Clicks (days ≤75)",
        "VLE clicks in the opening phase",
        0,
        10000,
        300,
    ),
    "total_clicks": (
        "Total VLE Clicks",
        "Total platform clicks across the whole module",
        0,
        50000,
        1500,
    ),
    "active_days": (
        "Active Days on VLE",
        "Number of distinct calendar days with at least one login",
        0,
        300,
        80,
    ),
    "total_sessions": (
        "Total VLE Sessions",
        "Total number of VLE login sessions",
        0,
        5000,
        200,
    ),
    "avg_clicks_per_session": (
        "Avg Clicks / Session",
        "Mean number of clicks per login session",
        0.0,
        100.0,
        7.0,
    ),
    "click_cv": (
        "Click Consistency (CV)",
        "Coefficient of variation of weekly clicks — lower = more regular",
        0.0,
        10.0,
        1.0,
    ),
    "cgpa_proxy": (
        "CGPA Proxy Score",
        "Weighted aggregate: 60% TMA average + 40% exam score (0-100 scale)",
        0.0,
        100.0,
        60.0,
    ),
    "exam_score": (
        "Exam Score",
        "Final exam score (enter 0 if exam not yet sat)",
        0.0,
        100.0,
        55.0,
    ),
    "tma_mean": (
        "TMA Mean Score",
        "Average score across all tutor-marked assignments",
        0.0,
        100.0,
        65.0,
    ),
    "tma_weighted_avg": (
        "TMA Weighted Average",
        "TMA scores weighted by each assignment's credit weight",
        0.0,
        100.0,
        65.0,
    ),
    "tma_max": (
        "TMA Maximum Score",
        "Highest score achieved on any single TMA",
        0.0,
        100.0,
        80.0,
    ),
    "tma_min": (
        "TMA Minimum Score",
        "Lowest score achieved on any single TMA",
        0.0,
        100.0,
        50.0,
    ),
    "tma_std": (
        "TMA Score Std Dev",
        "Standard deviation of TMA scores — lower = more consistent",
        0.0,
        50.0,
        10.0,
    ),
    "tma_score_trend": (
        "TMA Score Trend (slope)",
        "Linear slope of TMA scores over time — positive = improving",
        -50.0,
        50.0,
        0.0,
    ),
    "avg_days_early": (
        "Avg Days Early (submissions)",
        "Mean days before deadline that assignments were submitted",
        -30.0,
        30.0,
        2.0,
    ),
    "late_submission_rate": (
        "Late Submission Rate",
        "Proportion of TMAs submitted after the deadline (0–1)",
        0.0,
        1.0,
        0.1,
    ),
    "imd_band_numeric": (
        "Deprivation Index (IMD)",
        "Area-level deprivation 0–100; higher = less deprived",
        0,
        100,
        55,
    ),
    "edu_level_num": (
        "Highest Education Level",
        "Entry qualification (0=None … 4=Postgraduate)",
        0,
        4,
        2,
    ),
    "age_band_num": (
        "Age Band",
        "0 = Under 35  |  1 = 35–55  |  2 = 55 and over",
        0,
        2,
        0,
    ),
    "num_of_prev_attempts": (
        "Previous Module Attempts",
        "Number of prior attempts at this module",
        0,
        6,
        0,
    ),
    "studied_credits": (
        "Credits Being Studied",
        "Total credits registered for this period",
        0,
        600,
        60,
    ),
    "is_female": ("Gender", "Student gender", 0, 1, 0),
    "has_disability": (
        "Disability Declared",
        "Does the student have a declared disability?",
        0,
        1,
        0,
    ),
    "is_repeat_student": (
        "Repeat Student",
        "Has this student attempted this module before?",
        0,
        1,
        0,
    ),
}

INT_FEATS = {
    "clicks_late",
    "clicks_mid",
    "clicks_early",
    "total_clicks",
    "active_days",
    "total_sessions",
    "imd_band_numeric",
    "edu_level_num",
    "age_band_num",
    "num_of_prev_attempts",
    "studied_credits",
}

SIDEBAR_GROUPS = {
    "VLE Engagement": [
        "clicks_late",
        "clicks_mid",
        "clicks_early",
        "total_clicks",
        "active_days",
        "total_sessions",
        "avg_clicks_per_session",
        "click_cv",
    ],
    "Assessment": [
        "cgpa_proxy",
        "exam_score",
        "tma_mean",
        "tma_weighted_avg",
        "tma_max",
        "tma_min",
        "tma_std",
        "tma_score_trend",
        "avg_days_early",
        "late_submission_rate",
    ],
    "Demographics": [
        "imd_band_numeric",
        "edu_level_num",
        "age_band_num",
        "num_of_prev_attempts",
        "studied_credits",
        "is_female",
        "has_disability",
        "is_repeat_student",
    ],
}


# WIDGET FACTORY
def make_widget(feat):
    meta = FEAT_META.get(feat, (feat, feat, 0, 100, 50))
    label, tip, mn, mx, default = meta

    if feat == "is_female":
        v = st.selectbox("Gender", ["Male", "Female"], help="Student gender")
        return 1 if v == "Female" else 0
    if feat == "has_disability":
        v = st.selectbox("Disability Declared", ["No", "Yes"], help=tip)
        return 1 if v == "Yes" else 0
    if feat == "is_repeat_student":
        v = st.selectbox("Repeat Student", ["No", "Yes"], help=tip)
        return 1 if v == "Yes" else 0
    if feat == "edu_level_num":
        opts = [
            "No Formal Quals",
            "Lower Than A Level",
            "A Level or Equivalent",
            "HE Qualification",
            "Post Graduate Qualification",
        ]
        v = st.selectbox("Highest Education Level", opts, index=2, help=tip)
        return opts.index(v)
    if feat == "age_band_num":
        opts = ["Under 35", "35–55", "55 and over"]
        v = st.selectbox("Age Band", opts, index=0, help=tip)
        return opts.index(v)
    if feat in INT_FEATS:
        return st.number_input(label, int(mn), int(mx), int(default), help=tip)
    step = 0.01 if mx <= 1.0 else 0.5
    return st.number_input(
        label, float(mn), float(mx), float(default), step=step, help=tip
    )



# PREDICTION
def compute_cgpa(cgpa_proxy):
    """Convert UK marks (0-100) to CGPA scale (1.00-5.00)."""
    cgpa = round((cgpa_proxy / 100) * 4 + 1, 2)
    return max(1.00, min(5.00, cgpa))


def cgpa_to_degree_class(cgpa):
    """Map CGPA to degree classification."""
    if cgpa >= 4.50:
        return "First Class"
    elif cgpa >= 3.50:
        return "Second Class Upper"
    elif cgpa >= 2.40:
        return "Second Class Lower"
    elif cgpa >= 1.50:
        return "Third Class"
    else:
        return "Pass"


def run_prediction(inputs):
    pred_inputs = inputs.copy()
    
    if "cgpa" not in pred_inputs:
        pred_inputs["cgpa"] = compute_cgpa(pred_inputs.get("cgpa_proxy", 0))
        
    row = pd.DataFrame([pred_inputs])[TOP20]
    proba = model.predict_proba(row)[0]
    pred = int(np.argmax(proba))
    label = LABEL_MAP[str(pred)]
    conf = float(proba[pred])
    probs = {LABEL_MAP[str(i)]: float(proba[i]) for i in range(len(proba))}
    return label, conf, probs


# RISK FLAGS
def get_flags(inputs, pred):
    flags = []
    tc = inputs.get("total_clicks", 9999)
    ad = inputs.get("active_days", 9999)
    tm = inputs.get("tma_mean", 9999)
    ls = inputs.get("late_submission_rate", 0)
    tr = inputs.get("tma_score_trend", 0)
    es = inputs.get("exam_score", -1)
    pa = inputs.get("num_of_prev_attempts", 0)
    im = inputs.get("imd_band_numeric", 100)
    cgpa = compute_cgpa(inputs.get("cgpa_proxy", 0))

    if tc < 200:
        flags.append(
            (
                "box-risk",
                "🔴 <b>Very low VLE engagement.</b> Student may be disengaged from the platform entirely.",
            )
        )
    elif tc < 600:
        flags.append(
            (
                "box-warn",
                "🟡 <b>Below-average VLE engagement.</b> Monitor participation in upcoming weeks.",
            )
        )
    if ad < 15:
        flags.append(
            (
                "box-risk",
                "🔴 <b>Fewer than 15 active days on the VLE.</b> Very low attendance — consider welfare check.",
            )
        )
    if tm < 40:
        flags.append(
            (
                "box-risk",
                "🔴 <b>TMA mean below 40.</b> Student is consistently underperforming in coursework.",
            )
        )
    elif tm < 55:
        flags.append(
            (
                "box-warn",
                "🟡 <b>TMA mean below 55.</b> Borderline performance — targeted academic support recommended.",
            )
        )
    if ls > 0.5:
        flags.append(
            (
                "box-warn",
                "🟡 <b>Over 50% of TMAs submitted late.</b> May indicate personal difficulties or poor time management.",
            )
        )
    if tr < -5:
        flags.append(
            (
                "box-warn",
                "🟡 <b>Declining TMA score trend.</b> Performance is deteriorating — early intervention recommended.",
            )
        )
    elif tr > 5:
        flags.append(
            (
                "box-ok",
                "✅ <b>Improving TMA trend.</b> Student is showing positive score progression.",
            )
        )
    if 0 < es < 30:
        flags.append(
            (
                "box-risk",
                "🔴 <b>Low exam score detected.</b> Student is at high risk of poor final outcome.",
            )
        )
    if pa >= 2:
        flags.append(
            (
                "box-warn",
                f"🟡 <b>{pa} previous attempt(s) on this module.</b> Consider alternative support pathways.",
            )
        )
    if im < 20:
        flags.append(
            (
                "box-warn",
                "🟡 <b>Student from a highly deprived area (IMD &lt; 20%).</b> May benefit from bursary or pastoral referral.",
            )
        )
    if pred in ("Pass", "Third Class"):
        flags.append(
            (
                "box-risk",
                f"🔴 <b>Predicted {pred}.</b> Immediate academic advisor intervention is strongly recommended.",
            )
        )
    elif pred == "Second Class Lower":
        flags.append(
            (
                "box-warn",
                f"🟡 <b>Predicted {pred}.</b> Academic support recommended to improve outcomes.",
            )
        )
    return flags


# CHARTS
def prob_chart(probs):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")
    labels = CLASS_ORDER
    vals = [probs.get(l, 0) for l in labels]
    colors = [CLASS_COLOR[l] for l in labels]
    bars = ax.barh(
        labels, vals, color=colors, edgecolor="#0f1117", linewidth=0.6, height=0.5
    )
    for bar, val in zip(bars, vals):
        ax.text(
            min(val + 0.012, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_xlabel("Probability", color="#888", fontsize=9)
    ax.tick_params(colors="#ccc", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#2a2d3a")
    ax.set_title("Class Probabilities", color="#e0e0e0", fontsize=10, pad=8)
    plt.tight_layout()
    return fig


def importance_chart():
    labels = [FEAT_META.get(f, (f,))[0] for f in TOP20]
    group_color = {}
    for f in TOP20:
        if f in SIDEBAR_GROUPS["VLE Engagement"]:
            group_color[f] = "#4ef18a"
        elif f in SIDEBAR_GROUPS["Assessment"]:
            group_color[f] = "#4e9af1"
        else:
            group_color[f] = "#f1c94e"
    colors = [group_color[f] for f in TOP20]

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")
    ax.barh(
        labels[::-1],
        TOP20_IMP[::-1],
        color=colors[::-1],
        edgecolor="#0f1117",
        linewidth=0.4,
    )
    patches = [
        mpatches.Patch(color="#4ef18a", label="VLE Engagement"),
        mpatches.Patch(color="#4e9af1", label="Assessment"),
        mpatches.Patch(color="#f1c94e", label="Demographics"),
    ]
    ax.legend(
        handles=patches,
        fontsize=7.5,
        facecolor="#1a1d27",
        edgecolor="#2a2d3a",
        labelcolor="#ccc",
        loc="lower right",
    )
    ax.set_xlabel("RF Importance Score", color="#888", fontsize=9)
    ax.tick_params(colors="#ccc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a2d3a")
    ax.set_title("Top 20 Features Used by Model", color="#e0e0e0", fontsize=10, pad=8)
    plt.tight_layout()
    return fig


# LAYOUT
st.markdown(
    """
<div style='text-align:center;padding:18px 0 8px;'>
  <div style='font-size:2.6rem;font-weight:800;color:#4e9af1;'>
    🎓 Degree Class Prediction System
  </div>
  <div style='font-size:1rem;color:#888;margin-top:5px;'>
    CGPA-Based Degree Classification &nbsp;·&nbsp;
    Stacking Ensemble (RF + HGB + SVM → LR) &nbsp;·&nbsp; OULAD dataset
  </div>
</div>
<hr style='border-color:#2a2d3a;margin:6px 0 18px;'>
""",
    unsafe_allow_html=True,
)

if not LOADED:
    st.error(f"""
**Model files not found.** Run `implementation_degree_class.py` first to generate:
`stacking_top20_degree_class_model.pkl` · `top20_feature_names_degree_class.json` ·
`top20_feature_importances_degree_class.json` · `label_map_degree_class.json`

Missing file: `{ERR}`
""")
    st.stop()

# Sidebar
st.sidebar.markdown(
    '<div class="sec-hdr"> Model Performance (Test Set)</div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", "~75%", help="Hold-out test set accuracy")
c2.metric("Macro F1", "~0.70", help="Unweighted mean F1 across all 5 classes")
c3.metric("ROC-AUC", "~0.92", help="One-vs-Rest macro AUC")
c4.metric("Features", "20", help="Top-20 by Random Forest importance")

with st.sidebar.expander(" About this model", expanded=False):
    st.markdown("""
**Model:** Stacking Ensemble retrained on the **top-20 features** selected
by Random Forest native importance (Mean Decrease Impurity).

**Base learners:** Random Forest · Hist Gradient Boosting · SVM (RBF)

**Meta-learner:** Logistic Regression — learns the optimal combination
of each base learner's probability outputs.

**Training data:** OULAD — 32,593 student-module enrolments, 7 modules.

**Class imbalance:** SMOTE oversampling applied inside each CV fold.

**CGPA Formula:** CGPA = (cgpa_proxy / 100) × 4 + 1

**Degree Classes:**
🥇 First Class (CGPA 4.50–5.00) · 🥈 Second Class Upper (3.50–4.49) ·
🥉 Second Class Lower (2.40–3.49) · 📜 Third Class (1.50–2.39) · ✅ Pass (1.00–1.49)
    """)

st.sidebar.markdown(
    '<div class="sec-hdr"> Top 20 Feature Importances</div>',
    unsafe_allow_html=True,
)
st.sidebar.pyplot(importance_chart(), use_container_width=True)

# Main columns
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown("## Student Data Input")
    st.caption(
        "Enter available data for the student. For best results, fill in as many fields as possible. Use the default values as a reference point based on typical student profiles."
    )

    inputs = {}
    for group, feats in SIDEBAR_GROUPS.items():
        in_top20 = [f for f in feats if f in TOP20]
        if not in_top20:
            continue
        st.markdown(f"**{group}**")
        for feat in in_top20:
            inputs[feat] = make_widget(feat)
        
        st.markdown("---")

    predict_btn = st.button(
        "Predict Degree Class",
        type="primary",
        use_container_width=True,
    )

with right:
    st.markdown(
        '<div class="sec-hdr"> Prediction Output</div>', unsafe_allow_html=True
    )

    if not predict_btn:
        st.markdown(
            """
<div class="box-info">
 <b>Fill in student data</b>, then click
  <b>Predict Degree Class</b>.
</div>
<div class="box-info" style="margin-top:10px;">
  <b>Workflow for academic advisors:</b><br>
  1. Enter the student's current VLE engagement figures<br>
  2. Enter TMA scores and exam result (0 if not yet sat)<br>
  3. Fill in demographic background details<br>
  4. Click Predict and review the result + risk flags<br>
  5. Use the advisory flags to prioritise interventions
</div>
        """,
            unsafe_allow_html=True,
        )

    else:
        cgpa = compute_cgpa(inputs.get("cgpa_proxy", 0))
        cgpa_class = cgpa_to_degree_class(cgpa)
        pred, conf, probs = run_prediction(inputs)
        color = CLASS_COLOR[pred]
        emoji = CLASS_EMOJI[pred]
        card = CLASS_CARD[pred]

        # CGPA Display Card
        st.markdown(
            f"""
<div style="background:#1a1d27; border:1px solid #3a3d4d; border-radius:10px; padding:15px; margin-bottom:15px;">
  <div style="font-size:0.85rem; color:#888;">Computed CGPA (1.00 - 5.00)</div>
  <div style="font-size:1.8rem; font-weight:bold; color:{color};">{cgpa:.2f}</div>
  <div style="font-size:0.9rem; color:#888; margin-top:5px;">Direct CGPA Classification: <span style="color:{color};">{cgpa_class}</span></div>
</div>
        """,
            unsafe_allow_html=True,
        )

        # Prediction Result Card
        st.markdown(
            f"""
<div class="result-card {card}">
  <div class="res-emoji">{emoji}</div>
  <div class="res-label" style="color:{color};">{pred}</div>
  <div class="res-sub">Predicted Final Degree Class</div>
  <div class="res-conf" style="color:{color};">{conf*100:.1f}% confidence</div>
</div>
        """,
            unsafe_allow_html=True,
        )

        st.pyplot(prob_chart(probs), use_container_width=True)

        flags = get_flags(inputs, pred)
        if flags:
            st.markdown(
                '<div class="sec-hdr"> Advisory Flags</div>', unsafe_allow_html=True
            )
            for box_cls, msg in flags:
                st.markdown(
                    f'<div class="{box_cls}">{msg}</div>', unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
<div class="box-ok" style="margin-top:10px;">
   <b>No risk flags.</b> Student indicators are within healthy ranges.
</div>
            """,
                unsafe_allow_html=True,
            )

        with st.expander(" View submitted input values", expanded=False):
            rows = []
            for rank, feat in enumerate(TOP20, 1):
                if feat not in inputs:
                    continue
                rows.append(
                    {
                        "Rank": rank,
                        "Feature": FEAT_META.get(feat, (feat,))[0],
                        "Input Value": inputs[feat],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-hdr"> Export</div>', unsafe_allow_html=True)
        report_lines = [
            "OULAD Degree Class Prediction Report",
            "=" * 40,
            f"Computed CGPA      : {cgpa:.2f}",
            f"CGPA Classification : {cgpa_class}",
            f"Predicted Class    : {pred}",
            f"Confidence         : {conf*100:.1f}%",
            "",
            "Class Probabilities:",
        ]
        for cls in CLASS_ORDER:
            report_lines.append(f"  {cls:<22}: {probs[cls]*100:.1f}%")
        report_lines += ["", "Advisory Flags:"]
        for _, msg in flags:
            clean = msg.replace("<b>", "").replace("</b>", "")
            report_lines.append(f"  • {clean}")
        report_lines += ["", "Input Values:"]
        for feat in TOP20:
            if feat in inputs:
                lbl = FEAT_META.get(feat, (feat,))[0]
                report_lines.append(f"  {lbl}: {inputs[feat]}")

        st.download_button(
            label=" Download Prediction Report (.txt)",
            data="\n".join(report_lines),
            file_name="degree_class_prediction_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.markdown(
    """
<hr style='border-color:#2a2d3a;margin-top:40px;'>
<div style='text-align:center;color:#444;font-size:1rem;padding-bottom:16px;'>
  Project By: AKOH GRACE SAMUEL &nbsp;|&nbsp;
  MATRIC NUMBER: SU22202001T &nbsp;|&nbsp;
  Degree Class Prediction System &nbsp;|&nbsp;
  For academic advisory use only.
</div>
""",
    unsafe_allow_html=True,
)
