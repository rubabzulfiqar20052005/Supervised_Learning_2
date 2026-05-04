import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Clash+Display:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Bebas+Neue&display=swap');

*, html, body { font-family: 'DM Sans', sans-serif !important; }

.stApp { background: #05080f; }

section[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid #111d30;
}
section[data-testid="stSidebar"] * { color: #8090b0 !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label { color: #5a6888 !important; font-size:0.8rem !important; }

.hero {
    background: linear-gradient(160deg, #060d1e 0%, #0a1530 50%, #060d1e 100%);
    border: 1px solid #0f2040;
    border-radius: 20px;
    padding: 44px 52px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(239,68,68,0.05) 0%, transparent 65%);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 20%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(16,185,129,0.04) 0%, transparent 65%);
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem;
    font-weight: 700;
    color: #ef4444;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: block;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.8rem;
    color: #ffffff;
    line-height: 0.9;
    margin: 0 0 14px 0;
    letter-spacing: 1px;
}
.hero-title em {
    color: #ef4444;
    font-style: normal;
}
.hero-desc {
    color: #4a6080;
    font-size: 0.92rem;
    max-width: 560px;
    line-height: 1.65;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.25);
    color: #ef4444;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 4px;
    margin: 16px 4px 0 0;
    font-family: 'JetBrains Mono', monospace !important;
}
.hero-badge.green {
    background: rgba(16,185,129,0.1);
    border-color: rgba(16,185,129,0.25);
    color: #10b981;
}
.hero-badge.blue {
    background: rgba(59,130,246,0.1);
    border-color: rgba(59,130,246,0.25);
    color: #60a5fa;
}

.kpi-card {
    background: #080c18;
    border: 1px solid #111d30;
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.kpi-card:hover { border-color: #1e3050; transform: translateY(-2px); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #ef4444);
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent, #ef4444);
    line-height: 1;
    margin-bottom: 7px;
}
.kpi-lbl {
    font-size: 0.72rem;
    color: #334560;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 600;
}

.section-hd {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.7rem;
    color: #e8f0ff;
    letter-spacing: 0.5px;
    margin: 8px 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #0f2040;
}
.section-hd em { color: #ef4444; font-style: normal; }

.formula-box {
    background: #080c18;
    border: 1px solid #111d30;
    border-left: 3px solid #ef4444;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 14px 0;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem;
    color: #6a8aaa;
    line-height: 1.9;
}
.formula-box b { color: #c8d6f0; }

.match-pill {
    display: inline-block;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6ee7b7;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace !important;
    margin-left: 6px;
}

.stButton > button {
    background: linear-gradient(135deg, #dc2626, #ef4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
}

div[data-testid="stTabs"] button {
    color: #334560 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ef4444 !important;
    border-bottom-color: #ef4444 !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: #111d30; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────
BG    = "#05080f"
PAPER = "#080c18"
GRID  = "#0f1e35"
FONT  = "#4a6080"
RED   = "#ef4444"
GRN   = "#10b981"
BLU   = "#3b82f6"
YEL   = "#f59e0b"
PRP   = "#8b5cf6"
CYAN  = "#06b6d4"

def base_layout(title="", h=400):
    return dict(
        title=dict(text=title, font=dict(color="#c8d6f0", size=13, family="DM Sans")),
        plot_bgcolor=BG, paper_bgcolor=PAPER,
        font=dict(color=FONT, family="DM Sans"),
        height=h,
        margin=dict(l=55, r=25, t=50, b=50),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span style="font-family:JetBrains Mono;font-size:0.65rem;color:#334560;letter-spacing:2px;text-transform:uppercase;">⚙ CONFIGURATION</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    n_samples   = st.select_slider("Dataset Size", [2000,5000,10000,20000], value=10000)
    fraud_ratio = st.slider("Fraud Ratio (%)", 1, 15, 3, step=1)
    test_size   = st.slider("Test Split", 0.15, 0.40, 0.20, step=0.05)
    threshold   = st.slider("Active Threshold", 0.1, 0.9, 0.5, step=0.05)
    class_wt    = st.radio("Class Weight", ["balanced", "none"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔁  Retrain Model", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span style="font-family:JetBrains Mono;font-size:0.65rem;color:#334560;letter-spacing:2px;text-transform:uppercase;">Dataset Info</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:10px;font-size:0.8rem;color:#334560;line-height:2;">
    Samples: <span style="color:#6a8aaa">{n_samples}</span><br>
    Fraud: <span style="color:#ef4444">{fraud_ratio}%</span><br>
    Legit: <span style="color:#10b981">{100-fraud_ratio}%</span><br>
    Features: <span style="color:#6a8aaa">20</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <span class="hero-eyebrow">📊 Supervised Learning · Evaluation Metrics</span>
  <div class="hero-title">Metrics <em>Dashboard</em><br>Credit Fraud Detection</div>
  <p class="hero-desc">Imbalanced binary classification · Manual confusion matrix · ROC & Precision-Recall curves · Threshold sensitivity analysis</p>
  <div>
    <span class="hero-badge">Logistic Regression</span>
    <span class="hero-badge green">NumPy · sklearn</span>
    <span class="hero-badge blue">Threshold = {threshold}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA + MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_data
def run_pipeline(n, fraud_pct, ts, cw):
    fraud_w = fraud_pct / 100
    legit_w = 1.0 - fraud_w
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=15, n_redundant=5,
        weights=[legit_w, fraud_w], flip_y=0, random_state=42
    )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=ts, random_state=42, stratify=y)

    w = "balanced" if cw == "balanced" else None
    model = LogisticRegression(class_weight=w, max_iter=1000, random_state=42)
    model.fit(Xtr, ytr)

    y_proba = model.predict_proba(Xte)[:, 1]
    y_pred  = model.predict(Xte)

    fpr_arr, tpr_arr, roc_t = roc_curve(yte, y_proba)
    roc_auc_val = auc(fpr_arr, tpr_arr)
    prec_arr, rec_arr, pr_t = precision_recall_curve(yte, y_proba)
    ap = average_precision_score(yte, y_proba)

    return Xte, yte, y_pred, y_proba, fpr_arr, tpr_arr, roc_t, roc_auc_val, prec_arr, rec_arr, pr_t, ap

with st.spinner("Training model..."):
    Xte, yte, y_pred_50, y_proba, fpr_arr, tpr_arr, roc_t, roc_auc_val, prec_arr, rec_arr, pr_t, ap = run_pipeline(
        n_samples, fraud_ratio, test_size, class_wt
    )

# Apply user-selected threshold
y_pred_t = (y_proba >= threshold).astype(int)

# Manual confusion matrix
def manual_cm(yt, yp):
    TP = int(np.sum((yt==1)&(yp==1)))
    TN = int(np.sum((yt==0)&(yp==0)))
    FP = int(np.sum((yt==0)&(yp==1)))
    FN = int(np.sum((yt==1)&(yp==0)))
    return TP, TN, FP, FN

TP, TN, FP, FN = manual_cm(yte, y_pred_t)
PREC = TP/(TP+FP) if (TP+FP)>0 else 0.0
REC  = TP/(TP+FN) if (TP+FN)>0 else 0.0
F1   = 2*PREC*REC/(PREC+REC) if (PREC+REC)>0 else 0.0
ACC  = (TP+TN)/(TP+TN+FP+FN)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (f"{PREC:.3f}", "Precision",  RED,  "var(--accent,#ef4444)"),
    (f"{REC:.3f}",  "Recall",     GRN,  "var(--accent,#10b981)"),
    (f"{F1:.3f}",   "F1 Score",   BLU,  "var(--accent,#3b82f6)"),
    (f"{ACC:.3f}",  "Accuracy",   YEL,  "var(--accent,#f59e0b)"),
    (f"{roc_auc_val:.3f}", "ROC AUC", PRP, "var(--accent,#8b5cf6)"),
    (f"{ap:.3f}",   "Avg Prec",   CYAN, "var(--accent,#06b6d4)"),
]
for col, (val, lbl, clr, var) in zip([c1,c2,c3,c4,c5,c6], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:{clr}">
          <div class="kpi-val" style="color:{clr}">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋  Dataset",
    "🔢  Confusion Matrix",
    "📐  Manual Metrics",
    "📈  ROC & PR Curves",
    "🎚  Threshold Analysis",
    "🗂  Summary Table"
])


# ══════════════════════════════════════════════
# TAB 1 — Dataset
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hd">Dataset <em>Overview</em></div>', unsafe_allow_html=True)

    left, right = st.columns([1,1.4])

    fraud_n  = int(np.sum(yte==1))
    legit_n  = int(np.sum(yte==0))
    total_n  = len(yte)

    with left:
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[legit_n, fraud_n],
            hole=0.65,
            marker=dict(colors=[GRN, RED], line=dict(color=BG, width=3)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#e8f0ff"),
            pull=[0, 0.06]
        ))
        fig.update_layout(
            **base_layout("Class Distribution (Test Set)", h=320),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{total_n}</b><br>test<br>samples",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color="#e8f0ff")
            )]
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown(f"""
        <div style="margin-top:12px;">
          <div style="display:flex;gap:12px;margin-bottom:12px;">
            <div class="kpi-card" style="--accent:{GRN};flex:1">
              <div class="kpi-val" style="color:{GRN};font-size:1.6rem">{legit_n}</div>
              <div class="kpi-lbl">Legitimate</div>
            </div>
            <div class="kpi-card" style="--accent:{RED};flex:1">
              <div class="kpi-val" style="color:{RED};font-size:1.6rem">{fraud_n}</div>
              <div class="kpi-lbl">Fraud</div>
            </div>
          </div>
          <div class="formula-box">
            <b>Imbalance Ratio</b><br>
            {legit_n} : {fraud_n} = <b>{legit_n/max(fraud_n,1):.1f} : 1</b><br><br>
            <b>Why imbalance matters:</b><br>
            A naive model predicting all legitimate gets {legit_n/total_n*100:.1f}% accuracy but detects ZERO fraud. Accuracy alone is misleading — we need Precision, Recall & F1.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability distribution
        fraud_proba  = y_proba[yte==1]
        legit_proba  = y_proba[yte==0]

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=legit_proba, name="Legitimate", nbinsx=40,
                                    marker_color=GRN, opacity=0.7))
        fig2.add_trace(go.Histogram(x=fraud_proba, name="Fraud", nbinsx=40,
                                    marker_color=RED, opacity=0.7))
        fig2.add_vline(x=threshold, line=dict(color="white", dash="dot", width=1.5),
                       annotation_text=f"Threshold {threshold}",
                       annotation_font=dict(color="white", size=10))
        fig2.update_layout(
            **base_layout("Predicted Probability Distribution", h=250),
            barmode='overlay',
            legend=dict(orientation='h', y=1.1, font=dict(color="#c8d6f0")),
            margin=dict(l=50, r=20, t=50, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 2 — Confusion Matrix
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hd">Confusion <em>Matrix</em></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])

    with left:
        cm_vals = np.array([[TN, FP], [FN, TP]])
        labels  = [["TN", "FP"], ["FN", "TP"]]
        hover   = [[f"True Negatives<br>{TN}", f"False Positives<br>{FP}"],
                   [f"False Negatives<br>{FN}", f"True Positives<br>{TP}"]]

        text_m = [[f"<b>{TN}</b><br><span style='font-size:11px'>TN</span>",
                   f"<b>{FP}</b><br><span style='font-size:11px'>FP</span>"],
                  [f"<b>{FN}</b><br><span style='font-size:11px'>FN</span>",
                   f"<b>{TP}</b><br><span style='font-size:11px'>TP</span>"]]

        fig = go.Figure(go.Heatmap(
            z=cm_vals,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            colorscale=[[0, '#080c18'], [0.4,'#1a1040'], [1, RED]],
            showscale=False,
            text=text_m,
            texttemplate="%{text}",
            textfont=dict(size=22, color="white"),
            hoverinfo='skip'
        ))
        fig.update_layout(
            **base_layout(f"Confusion Matrix  (Threshold = {threshold})", h=360),
            xaxis=dict(side='bottom', tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=110, r=20, t=55, b=80)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown(f"""
        <div class="formula-box" style="margin-top:0">
          <b>Confusion Matrix Breakdown</b><br><br>
          <b style="color:{GRN}">TN = {TN}</b> — Correctly predicted Legitimate<br>
          <b style="color:{RED}">TP = {TP}</b> — Correctly predicted Fraud<br>
          <b style="color:{YEL}">FP = {FP}</b> — Legit flagged as Fraud (Type I Error)<br>
          <b style="color:#f87171">FN = {FN}</b> — Fraud missed (Type II Error)<br><br>
          <b>Total test samples:</b> {TP+TN+FP+FN}<br>
          <b>Correctly classified:</b> {TP+TN} ({(TP+TN)/(TP+TN+FP+FN)*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)

        # Visual TP/TN/FP/FN bars
        fig2 = go.Figure(go.Bar(
            x=["TN", "TP", "FP", "FN"],
            y=[TN, TP, FP, FN],
            marker_color=[GRN, RED, YEL, "#f87171"],
            text=[TN, TP, FP, FN],
            textposition='outside',
            textfont=dict(color="#e8f0ff", size=13),
            width=0.5
        ))
        fig2.update_layout(
            **base_layout("Cell Values", h=280),
            margin=dict(l=30, r=20, t=50, b=40),
            yaxis=dict(gridcolor=GRID)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 3 — Manual Metrics
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hd">Manual <em>Metrics</em> — NumPy vs sklearn</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="formula-box">
      <b>Precision</b>  = TP / (TP + FP) = {TP} / ({TP} + {FP}) = <b style="color:{RED}">{PREC:.4f}</b><br>
      <b>Recall</b>     = TP / (TP + FN) = {TP} / ({TP} + {FN}) = <b style="color:{GRN}">{REC:.4f}</b><br>
      <b>F1 Score</b>   = 2 × P × R / (P + R)  = 2 × {PREC:.3f} × {REC:.3f} / ({PREC:.3f} + {REC:.3f}) = <b style="color:{BLU}">{F1:.4f}</b><br>
      <b>Accuracy</b>   = (TP + TN) / N  = ({TP} + {TN}) / {TP+TN+FP+FN} = <b style="color:{YEL}">{ACC:.4f}</b>
    </div>
    """, unsafe_allow_html=True)

    # Verification table
    sk_p = precision_score(yte, y_pred_t, zero_division=0)
    sk_r = recall_score(yte, y_pred_t, zero_division=0)
    sk_f = f1_score(yte, y_pred_t, zero_division=0)
    sk_a = accuracy_score(yte, y_pred_t)

    df_verify = pd.DataFrame({
        "Metric":         ["Precision", "Recall", "F1 Score", "Accuracy"],
        "Manual (NumPy)": [PREC, REC, F1, ACC],
        "sklearn":        [sk_p, sk_r, sk_f, sk_a],
        "Match":          ["✅" if np.isclose(a,b,atol=1e-6) else "❌"
                           for a,b in [(PREC,sk_p),(REC,sk_r),(F1,sk_f),(ACC,sk_a)]]
    })
    st.dataframe(
        df_verify.style
            .format({"Manual (NumPy)": "{:.6f}", "sklearn": "{:.6f}"})
            .background_gradient(cmap='Reds', subset=["Manual (NumPy)", "sklearn"]),
        use_container_width=True, hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart
    categories = ["Precision", "Recall", "F1 Score", "Accuracy", "ROC AUC"]
    vals_manual = [PREC, REC, F1, ACC, roc_auc_val]
    vals_sklearn = [sk_p, sk_r, sk_f, sk_a, roc_auc_val]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_manual + [vals_manual[0]],
        theta=categories + [categories[0]],
        fill='toself', fillcolor=f'rgba(239,68,68,0.12)',
        line=dict(color=RED, width=2),
        name="Manual (NumPy)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_sklearn + [vals_sklearn[0]],
        theta=categories + [categories[0]],
        fill='toself', fillcolor=f'rgba(16,185,129,0.08)',
        line=dict(color=GRN, width=2, dash='dot'),
        name="sklearn"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID, color=FONT, tickfont=dict(size=9)),
            angularaxis=dict(color="#c8d6f0")
        ),
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        height=400,
        legend=dict(font=dict(color="#c8d6f0"), bgcolor=PAPER),
        margin=dict(l=60,r=60,t=40,b=40),
        title=dict(text="Manual vs sklearn Metrics (Radar)", font=dict(color="#c8d6f0", size=13))
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 4 — ROC & PR
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hd">ROC & <em>Precision-Recall</em> Curves</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        # ROC
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr, fill='tozeroy',
            fillcolor=f'rgba(239,68,68,0.08)',
            line=dict(color=RED, width=2.5),
            name=f"ROC (AUC={roc_auc_val:.4f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(color='white', width=1, dash='dot'),
            name='Random (AUC=0.5)', hoverinfo='skip'
        ))
        # Mark selected threshold
        idx = np.argmin(np.abs(roc_t - threshold))
        fig.add_trace(go.Scatter(
            x=[fpr_arr[idx]], y=[tpr_arr[idx]], mode='markers',
            marker=dict(color=YEL, size=12, symbol='diamond',
                        line=dict(color='white', width=1.5)),
            name=f"Threshold {threshold}"
        ))
        fig.update_layout(
            **base_layout(f"ROC Curve  (AUC = {roc_auc_val:.4f})", h=420),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(orientation='h', y=1.12, font=dict(color="#c8d6f0", size=10)),
            xaxis=dict(range=[0,1], gridcolor=GRID),
            yaxis=dict(range=[0,1.02], gridcolor=GRID),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # PR Curve
        baseline = np.sum(yte==1)/len(yte)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rec_arr, y=prec_arr, fill='tozeroy',
            fillcolor='rgba(139,92,246,0.08)',
            line=dict(color=PRP, width=2.5),
            name=f"PR (AP={ap:.4f})",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
        ))
        fig2.add_hline(y=baseline, line=dict(color='white', dash='dot', width=1),
                       annotation_text=f"Baseline ({baseline:.3f})",
                       annotation_font=dict(color='white', size=10))
        idx_pr = np.argmin(np.abs(pr_t - threshold)) if len(pr_t) > 0 else 0
        fig2.add_trace(go.Scatter(
            x=[rec_arr[idx_pr]], y=[prec_arr[idx_pr]], mode='markers',
            marker=dict(color=YEL, size=12, symbol='diamond',
                        line=dict(color='white', width=1.5)),
            name=f"Threshold {threshold}"
        ))
        fig2.update_layout(
            **base_layout(f"Precision-Recall Curve  (AP = {ap:.4f})", h=420),
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(orientation='h', y=1.12, font=dict(color="#c8d6f0", size=10)),
            xaxis=dict(range=[0,1], gridcolor=GRID),
            yaxis=dict(range=[0,1.05], gridcolor=GRID),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div class="formula-box">
      <b>ROC AUC</b> measures the probability that the model ranks a random positive sample higher than a random negative. AUC = <b style="color:{RED}">{roc_auc_val:.4f}</b> means {roc_auc_val*100:.1f}% of the time, the model correctly separates fraud from legit.<br><br>
      <b>PR Curve Average Precision</b> is more informative for imbalanced data. AP = <b style="color:{PRP}">{ap:.4f}</b> vs random baseline of {baseline:.3f}.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — Threshold Analysis
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-hd">Threshold <em>Sensitivity</em> Analysis (0.1 → 0.9)</div>', unsafe_allow_html=True)

    thresholds = np.arange(0.05, 0.95, 0.02)
    precs, recs, f1s, accs = [], [], [], []

    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        tp = np.sum((yte==1)&(yp==1))
        tn = np.sum((yte==0)&(yp==0))
        fp = np.sum((yte==0)&(yp==1))
        fn = np.sum((yte==1)&(yp==0))
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        precs.append(p); recs.append(r); f1s.append(f1)
        accs.append((tp+tn)/(tp+tn+fp+fn))

    fig = go.Figure()
    traces = [
        (precs, "Precision", RED),
        (recs,  "Recall",    GRN),
        (f1s,   "F1 Score",  BLU),
        (accs,  "Accuracy",  YEL),
    ]
    for vals, name, clr in traces:
        fig.add_trace(go.Scatter(
            x=thresholds, y=vals,
            name=name, line=dict(color=clr, width=2.5),
            mode='lines',
            hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>"
        ))

    # Shade tradeoff zone between precision and recall
    fig.add_trace(go.Scatter(
        x=list(thresholds)+list(thresholds[::-1]),
        y=list(precs)+list(recs[::-1]),
        fill='toself', fillcolor='rgba(255,255,255,0.02)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
    ))

    # Vertical line at selected threshold
    fig.add_vline(
        x=threshold,
        line=dict(color=YEL, dash='dot', width=2),
        annotation_text=f"  Active Threshold = {threshold}",
        annotation_font=dict(color=YEL, size=11),
        annotation_position="top left"
    )

    # Find crossover (P=R)
    diff_pr = np.abs(np.array(precs) - np.array(recs))
    cross_idx = np.argmin(diff_pr)
    fig.add_trace(go.Scatter(
        x=[thresholds[cross_idx]], y=[precs[cross_idx]],
        mode='markers+text',
        marker=dict(color='white', size=10),
        text=[f"P=R≈{precs[cross_idx]:.2f}"],
        textposition='top right',
        textfont=dict(color='white', size=10),
        name='P=R Crossover', showlegend=False
    ))

    fig.update_layout(
        **base_layout("Precision, Recall, F1, Accuracy vs Threshold", h=450),
        xaxis_title="Classification Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.08], gridcolor=GRID),
        xaxis=dict(range=[0.04, 0.96], gridcolor=GRID),
        legend=dict(orientation='h', y=1.1, font=dict(color="#c8d6f0")),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Table at selected thresholds
    st.markdown('<div class="section-hd" style="font-size:1.2rem;margin-top:4px;">Key <em>Threshold</em> Comparison</div>', unsafe_allow_html=True)
    key_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for t in key_thresholds:
        yp = (y_proba >= t).astype(int)
        tp = np.sum((yte==1)&(yp==1)); tn = np.sum((yte==0)&(yp==0))
        fp = np.sum((yte==0)&(yp==1)); fn = np.sum((yte==1)&(yp==0))
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        ac = (tp+tn)/(tp+tn+fp+fn)
        rows.append({"Threshold": f"{t:.1f}", "TP":tp,"TN":tn,"FP":fp,"FN":fn,
                     "Precision":round(p,4),"Recall":round(r,4),
                     "F1":round(f1,4),"Accuracy":round(ac,4),
                     "Active": "◀" if abs(t-threshold)<0.01 else ""})
    df_thresh = pd.DataFrame(rows)
    st.dataframe(
        df_thresh.style.background_gradient(cmap='Reds', subset=['F1']),
        use_container_width=True, hide_index=True
    )

    st.markdown(f"""
    <div class="formula-box">
      <b>Threshold Tradeoff Insight:</b><br>
      Lowering threshold → higher Recall (catch more fraud) but lower Precision (more false alarms).<br>
      Raising threshold → higher Precision (fewer false alarms) but lower Recall (miss more fraud).<br>
      F1 Score balances both — peaks at the optimal threshold.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 6 — Summary Table
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-hd">Final Metrics <em>Summary</em></div>', unsafe_allow_html=True)

    # sklearn classification report
    from io import StringIO
    cr = classification_report(yte, y_pred_t, target_names=["Legitimate","Fraud"])

    st.markdown(f"""
    <div class="formula-box">
      <b>sklearn classification_report() — Threshold = {threshold}</b><br><br>
      <pre style="color:#8aaccc;margin:0;font-family:'JetBrains Mono',monospace;font-size:0.8rem">{cr}</pre>
    </div>
    """, unsafe_allow_html=True)

    # Full summary df
    summary_data = {
        "Metric":         ["Precision","Recall","F1 Score","Accuracy","ROC AUC","Avg Precision (PR-AUC)"],
        "Manual (NumPy)": [f"{PREC:.6f}", f"{REC:.6f}", f"{F1:.6f}", f"{ACC:.6f}", "—", "—"],
        "sklearn":        [f"{sk_p:.6f}", f"{sk_r:.6f}", f"{sk_f:.6f}", f"{sk_a:.6f}",
                           f"{roc_auc_val:.6f}", f"{ap:.6f}"],
        "Interpretation": [
            "Of predicted fraud, what % is real fraud",
            "Of all fraud, what % was caught",
            "Harmonic mean of Precision & Recall",
            "Overall correct predictions",
            "Probability of correct ranking",
            "Weighted mean precision over all thresholds"
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # Gauge charts
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (val, lbl, clr) in zip(cols, [
        (PREC, "Precision",  RED),
        (REC,  "Recall",     GRN),
        (F1,   "F1 Score",   BLU),
    ]):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number=dict(font=dict(color=clr, size=28), valueformat=".3f"),
                title=dict(text=lbl, font=dict(color="#c8d6f0", size=13)),
                gauge=dict(
                    axis=dict(range=[0,1], tickfont=dict(color=FONT)),
                    bar=dict(color=clr, thickness=0.25),
                    bgcolor=GRID,
                    bordercolor=GRID,
                    steps=[
                        dict(range=[0, 0.5], color='rgba(239,68,68,0.08)'),
                        dict(range=[0.5, 0.75], color='rgba(245,158,11,0.08)'),
                        dict(range=[0.75, 1], color='rgba(16,185,129,0.08)'),
                    ],
                    threshold=dict(line=dict(color="white", width=1.5),
                                   thickness=0.8, value=val)
                )
            ))
            fig.update_layout(
                paper_bgcolor=PAPER, plot_bgcolor=BG,
                height=220, margin=dict(l=20,r=20,t=40,b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px;border-top:1px solid #0f2040;margin-top:20px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#1e3050;letter-spacing:2px;">
    METRICS DASHBOARD  ·  CREDIT FRAUD DETECTION  ·  ML COURSE  ·  BSCS 6TH SEMESTER
  </span>
</div>
""", unsafe_allow_html=True)
