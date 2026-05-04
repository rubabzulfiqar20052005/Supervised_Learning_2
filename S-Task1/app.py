import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Logistic Regression Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

*, html, body {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #080c14;
}

section[data-testid="stSidebar"] {
    background: #0d1320 !important;
    border-right: 1px solid #1e2d4a;
}

section[data-testid="stSidebar"] * {
    color: #c8d6f0 !important;
}

.hero-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0f2040 40%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px;
    right: -80px;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px;
    left: 30%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}

.hero-title span {
    background: linear-gradient(90deg, #00d4ff, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 1rem;
    color: #7a90b0;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.hero-tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00d4ff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
    font-family: 'Space Mono', monospace !important;
}

.metric-card {
    background: #0d1929;
    border: 1px solid #1a2e4a;
    border-radius: 12px;
    padding: 22px 24px;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: #2a4a7a;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'Space Mono', monospace !important;
    line-height: 1;
    margin-bottom: 6px;
}

.metric-label {
    font-size: 0.78rem;
    color: #5a7090;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8f0ff;
    margin: 10px 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1a2e4a;
    letter-spacing: -0.3px;
}

.section-title span {
    color: #00d4ff;
}

.info-box {
    background: rgba(0, 212, 255, 0.05);
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 16px 0;
    font-size: 0.9rem;
    color: #8aaccc;
    line-height: 1.6;
}

.pill-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.5px;
}

.pill-badge.green {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.3);
    color: #6ee7b7;
}

.pill-badge.red {
    background: rgba(239,68,68,0.12);
    border-color: rgba(239,68,68,0.3);
    color: #fca5a5;
}

div[data-testid="stSlider"] label {
    color: #7a90b0 !important;
    font-size: 0.85rem !important;
}

div[data-testid="stSlider"] div[data-baseweb="slider"] div {
    background: #1e3a5f !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0070f3, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 28px !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stSelectbox label, .stSlider label, .stRadio label {
    color: #7a90b0 !important;
}

div[data-testid="stTabs"] button {
    color: #5a7090 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom-color: #00d4ff !important;
}

.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
}

h1, h2, h3, h4, p {
    color: #c8d6f0;
}

.weight-row-manual { color: #00d4ff; }
.weight-row-sklearn { color: #a5b4fc; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1320; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

.stSpinner > div {
    border-top-color: #00d4ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOT_BG   = "#080c14"
PAPER_BG  = "#0d1929"
GRID_CLR  = "#1a2e4a"
FONT_CLR  = "#8aaccc"
ACCENT1   = "#00d4ff"
ACCENT2   = "#6366f1"
ACCENT3   = "#10b981"
ACCENT4   = "#f59e0b"

def base_layout(title="", h=420):
    return dict(
        title=dict(text=title, font=dict(color="#e8f0ff", size=14, family="Syne")),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, family="Syne"),
        height=h,
    )

# ─────────────────────────────────────────────
# MANUAL MODEL CLASS
# ─────────────────────────────────────────────
class LogisticRegressionManual:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        n, f = X.shape
        self.weights = np.zeros(f)
        self.bias = 0.0
        self.loss_history = []
        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            yp = self.sigmoid(z)
            self.loss_history.append(self.compute_loss(y, yp))
            e = yp - y
            self.weights -= self.lr * (X.T @ e) / n
            self.bias    -= self.lr * e.mean()

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-tag">⚙ HYPERPARAMETERS</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        value=0.1
    )
    epochs = st.slider("Training Epochs", 100, 2000, 1000, step=100)
    test_size = st.slider("Test Split Size", 0.10, 0.40, 0.20, step=0.05)
    feat1 = st.selectbox("Decision Boundary Feature 1", [0,1,2,3,4,5,6,7,8,9], index=0,
                         format_func=lambda i: load_breast_cancer().feature_names[i])
    feat2 = st.selectbox("Decision Boundary Feature 2", [0,1,2,3,4,5,6,7,8,9], index=1,
                         format_func=lambda i: load_breast_cancer().feature_names[i])

    st.markdown("<br>", unsafe_allow_html=True)
    train_btn = st.button("🚀  Train All Models", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;color:#3a5070;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Datasets Used</div>', unsafe_allow_html=True)
    st.markdown('<span class="pill-badge">Breast Cancer Wisconsin</span><span class="pill-badge">Iris</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;color:#3a5070;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Libraries</div>', unsafe_allow_html=True)
    st.markdown('<span class="pill-badge green">NumPy</span><span class="pill-badge green">sklearn</span><span class="pill-badge green">Plotly</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-tag">Machine Learning Lab</div>
  <div class="hero-title">Logistic Regression <span>from Scratch</span></div>
  <p class="hero-sub">Binary classification with NumPy  ·  sklearn comparison  ·  OvR multiclass  ·  Interactive decision boundaries</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD + PROCESS DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def prepare_data(test_sz, lr, ep):
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=test_sz, random_state=42)

    manual = LogisticRegressionManual(lr=lr, epochs=ep)
    manual.fit(Xtr, ytr)

    sk = LogisticRegression(max_iter=2000, random_state=42)
    sk.fit(Xtr, ytr)

    iris = load_iris()
    Xi, yi = iris.data, iris.target
    sci = StandardScaler()
    Xis = sci.fit_transform(Xi)
    Xitr, Xite, yitr, yite = train_test_split(Xis, yi, test_size=test_sz, random_state=42)
    from sklearn.multiclass import OneVsRestClassifier
    ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=42))
    ovr.fit(Xitr, yitr)

    return (data, X, y, Xs, Xtr, Xte, ytr, yte, manual, sk,
            iris, Xi, yi, Xis, Xitr, Xite, yitr, yite, ovr)


with st.spinner("Training models..."):
    (data, X, y, Xs, Xtr, Xte, ytr, yte, manual, sk,
     iris, Xi, yi, Xis, Xitr, Xite, yitr, yite, ovr) = prepare_data(test_size, learning_rate, epochs)

acc_m  = accuracy_score(yte, manual.predict(Xte))
acc_sk = accuracy_score(yte, sk.predict(Xte))
acc_ov = accuracy_score(yite, ovr.predict(Xite))
final_loss = manual.loss_history[-1]


# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    (f"{acc_m*100:.1f}%",   "Manual LR Accuracy"),
    (f"{acc_sk*100:.1f}%",  "sklearn LR Accuracy"),
    (f"{acc_ov*100:.1f}%",  "OvR Iris Accuracy"),
    (f"{final_loss:.4f}",   "Final Log-Loss"),
]
for col, (val, lbl) in zip([c1,c2,c3,c4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Dataset Overview",
    "🔧  Manual vs sklearn",
    "📉  Loss Curve",
    "🗺  Decision Boundary",
    "🌸  Iris OvR"
])


# ══════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Breast Cancer <span>Wisconsin</span> Dataset</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        # Class distribution donut
        malignant = int(np.sum(y == 0))
        benign    = int(np.sum(y == 1))
        fig = go.Figure(go.Pie(
            labels=["Malignant", "Benign"],
            values=[malignant, benign],
            hole=0.62,
            marker=dict(colors=[ACCENT2, ACCENT1],
                        line=dict(color=PLOT_BG, width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#e8f0ff"),
        ))
        fig.update_layout(
            **base_layout("Class Distribution", h=320),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{len(y)}</b><br><span style='font-size:11px'>samples</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#e8f0ff")
            )],
            margin=dict(l=55, r=25, t=50, b=50),
            xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
            yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<div class="info-box">🔬 <b>569 samples</b>, 30 numerical features computed from digitized breast mass images. Binary classification: <b>Malignant (0)</b> vs <b>Benign (1)</b>.</div>', unsafe_allow_html=True)

        fig2 = go.Figure(go.Bar(
            x=[malignant, benign],
            y=["Malignant", "Benign"],
            orientation='h',
            marker=dict(
                color=[ACCENT2, ACCENT1],
                line=dict(color=PLOT_BG, width=1)
            ),
            text=[malignant, benign],
            textposition='outside',
            textfont=dict(color="#e8f0ff", size=13)
        ))
        fig2.update_layout(**base_layout("Sample Count per Class", h=200),
                      margin=dict(l=80, r=60, t=50, b=30),
                      xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
                      yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Feature stats
    st.markdown('<div class="section-title" style="margin-top:10px;">Feature <span>Statistics</span> (first 10)</div>', unsafe_allow_html=True)
    df_feat = pd.DataFrame(Xs[:, :10], columns=data.feature_names[:10])
    desc = df_feat.describe().T[['mean','std','min','max']].round(3)
    desc.columns = ['Mean', 'Std Dev', 'Min', 'Max']
    st.dataframe(
        desc.style
            .background_gradient(cmap='Blues', subset=['Std Dev'])
            .format("{:.3f}"),
        use_container_width=True
    )


# ══════════════════════════════════════════════
# TAB 2 — Manual vs sklearn weights
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Weight Comparison: <span>Manual vs sklearn</span></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    feat_names = list(data.feature_names[:15])
    m_w  = manual.weights[:15]
    sk_w = sk.coef_[0][:15]
    diff = m_w - sk_w

    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Manual (NumPy)",
            x=feat_names,
            y=m_w,
            marker_color=ACCENT1,
            opacity=0.85
        ))
        fig.add_trace(go.Bar(
            name="sklearn",
            x=feat_names,
            y=sk_w,
            marker_color=ACCENT2,
            opacity=0.85
        ))
        fig.update_layout(
            **base_layout("Learned Weights — First 15 Features", h=380),
            barmode='group',
            legend=dict(orientation='h', y=1.12, x=0,
                        font=dict(color="#c8d6f0")),
            xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR, tickangle=-35, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
            margin=dict(l=55, r=25, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        fig2 = go.Figure(go.Bar(
            x=feat_names,
            y=np.abs(diff),
            marker=dict(
                color=np.abs(diff),
                colorscale=[[0,'#0d1929'],[0.5,'#6366f1'],[1,'#00d4ff']],
                showscale=False
            ),
            text=[f"{d:.3f}" for d in diff],
            textposition='outside',
            textfont=dict(size=9, color="#8aaccc")
        ))
        fig2.update_layout(
            **base_layout("|Manual − sklearn| Absolute Difference", h=380),
            xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR, tickangle=-35, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
            margin=dict(l=55, r=25, t=50, b=50)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="info-box">💡 Difference arises because sklearn applies <b>L2 Regularization (C=1.0)</b> by default, which penalizes large weights. Our manual model has no regularization, so weights can grow freely.</div>', unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-title" style="margin-top:6px;">Detailed <span>Weight Table</span></div>', unsafe_allow_html=True)
    df_w = pd.DataFrame({
        "Feature": data.feature_names[:15],
        "Manual Weight": m_w.round(6),
        "sklearn Weight": sk_w.round(6),
        "Absolute Diff": np.abs(diff).round(6)
    })
    st.dataframe(
        df_w.style
            .background_gradient(cmap='Blues', subset=['Absolute Diff'])
            .format({"Manual Weight": "{:.6f}", "sklearn Weight": "{:.6f}", "Absolute Diff": "{:.6f}"}),
        use_container_width=True
    )

    # Bias
    c1, c2, _ = st.columns([1,1,2])
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;">{manual.bias:.4f}</div>
            <div class="metric-label">Manual Bias</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;color:#a5b4fc;">{sk.intercept_[0]:.4f}</div>
            <div class="metric-label">sklearn Bias</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — Loss Curve
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Training <span>Loss Curve</span> — Binary Cross Entropy</div>', unsafe_allow_html=True)

    loss_arr = np.array(manual.loss_history)
    ep_arr   = np.arange(1, len(loss_arr)+1)

    # Fill under curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ep_arr, y=loss_arr,
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.05)',
        line=dict(color=ACCENT1, width=2.5),
        name="Log-Loss",
        hovertemplate="Epoch %{x}<br>Loss: %{y:.6f}<extra></extra>"
    ))
    # Mark convergence region
    conv_start = int(len(loss_arr) * 0.6)
    fig.add_vrect(
        x0=ep_arr[conv_start], x1=ep_arr[-1],
        fillcolor="rgba(99,102,241,0.05)",
        line=dict(color="rgba(99,102,241,0.2)", width=1),
        annotation_text="Convergence Zone",
        annotation_position="top right",
        annotation_font=dict(color="#a5b4fc", size=11)
    )
    fig.update_layout(
        **base_layout(f"Log-Loss over {epochs} Epochs  |  LR = {learning_rate}", h=400),
        xaxis_title="Epoch",
        yaxis_title="Binary Cross-Entropy Loss",
        hovermode='x unified',
        margin=dict(l=55, r=25, t=50, b=50),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Loss stats
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        (f"{loss_arr[0]:.4f}",  "Initial Loss"),
        (f"{loss_arr[-1]:.4f}", "Final Loss"),
        (f"{loss_arr.min():.4f}","Min Loss"),
        (f"{(loss_arr[0]-loss_arr[-1]):.4f}", "Total Drop"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.5rem;">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="info-box">📐 <b>Log-Loss Formula:</b>  L = −(1/N) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]  |  Gradient Descent updates: w = w − α·(∂L/∂w)</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — Decision Boundary
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Decision <span>Boundary</span> — Two Selected Features</div>', unsafe_allow_html=True)

    fi = feat1
    fj = feat2

    if fi == fj:
        st.warning("Please select two different features from the sidebar.")
    else:
        fl = [data.feature_names[fi], data.feature_names[fj]]
        X2d = Xs[:, [fi, fj]]
        Xtr2, Xte2, ytr2, yte2 = train_test_split(X2d, y, test_size=test_size, random_state=42)

        m2 = LogisticRegressionManual(lr=learning_rate, epochs=epochs)
        m2.fit(Xtr2, ytr2)
        s2 = LogisticRegression(max_iter=2000, random_state=42)
        s2.fit(Xtr2, ytr2)

        # Mesh
        margin = 0.7
        x_min, x_max = X2d[:,0].min()-margin, X2d[:,0].max()+margin
        y_min, y_max = X2d[:,1].min()-margin, X2d[:,1].max()+margin
        res = 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                              np.linspace(y_min, y_max, res))
        grid = np.c_[xx.ravel(), yy.ravel()]

        Zm = m2.predict_proba(grid)[:, 1].reshape(xx.shape)
        Zs = s2.predict_proba(grid)[:, 1].reshape(xx.shape)

        colors_0 = ["Malignant (0)", "Benign (1)"]
        pt_colors = [ACCENT2 if yi==0 else ACCENT1 for yi in y]

        left, right = st.columns(2)

        for col, Z, title, model in zip(
            [left, right],
            [Zm, Zs],
            ["Manual (NumPy) — Decision Boundary", "sklearn — Decision Boundary"],
            [m2, s2]
        ):
            with col:
                acc_2d = accuracy_score(yte2, model.predict(Xte2))
                fig = go.Figure()
                fig.add_trace(go.Contour(
                    z=Z, x=np.linspace(x_min, x_max, res),
                    y=np.linspace(y_min, y_max, res),
                    colorscale=[[0,'rgba(99,102,241,0.35)'],
                                [0.5,'rgba(15,32,64,0.1)'],
                                [1,'rgba(0,212,255,0.35)']],
                    showscale=False,
                    contours=dict(start=0, end=1, size=0.1),
                    line=dict(width=0),
                    hoverinfo='skip',
                ))
                # Decision line at 0.5
                fig.add_trace(go.Contour(
                    z=Z, x=np.linspace(x_min, x_max, res),
                    y=np.linspace(y_min, y_max, res),
                    showscale=False,
                    contours=dict(start=0.5, end=0.5, size=0),
                    contours_coloring='lines',
                    line=dict(color='white', width=1.5, dash='dot'),
                    hoverinfo='skip',
                    name="Decision Line"
                ))
                # Data points
                for cls, clr, lbl in [(0, ACCENT2, 'Malignant'), (1, ACCENT1, 'Benign')]:
                    mask = y == cls
                    fig.add_trace(go.Scatter(
                        x=X2d[mask, 0], y=X2d[mask, 1],
                        mode='markers',
                        marker=dict(color=clr, size=5, opacity=0.7,
                                    line=dict(color='white', width=0.4)),
                        name=lbl
                    ))
                fig.update_layout(
                    **base_layout(f"{title}<br><sup>Accuracy on 2-feature test set: {acc_2d*100:.1f}%</sup>", h=380),
                    xaxis_title=fl[0],
                    yaxis_title=fl[1],
                    legend=dict(orientation='h', y=1.15, font=dict(color="#c8d6f0", size=10)),
                    margin=dict(l=55, r=25, t=50, b=50),
                    xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
                    yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR)
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f'<div class="info-box">🗺 Showing decision boundary on: <b>{fl[0]}</b> vs <b>{fl[1]}</b>. The dotted white line is the decision boundary at probability = 0.5. Change features from the sidebar to explore.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — Iris OvR
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Iris Dataset — <span>OvR Multiclass</span> Classification</div>', unsafe_allow_html=True)

    proba = ovr.predict_proba(Xite)
    ypred = ovr.predict(Xite)
    cls_names = iris.target_names

    st.markdown('<div class="info-box">🌸 <b>One-vs-Rest (OvR) Strategy:</b> 3 binary classifiers trained — Setosa vs rest, Versicolor vs rest, Virginica vs rest. Final prediction = class with highest probability. Total accuracy: <b>' + f'{acc_ov*100:.2f}%</b></div>', unsafe_allow_html=True)

    top, _ = st.columns([3, 1])
    with top:
        # Grouped bar — predict_proba for first 20 samples
        n_show = min(20, len(proba))
        sample_ids = [f"S{i+1}" for i in range(n_show)]
        colors_cls = [ACCENT1, ACCENT2, ACCENT3]

        fig = go.Figure()
        for i, (cls, clr) in enumerate(zip(cls_names, colors_cls)):
            fig.add_trace(go.Bar(
                name=cls.capitalize(),
                x=sample_ids,
                y=proba[:n_show, i],
                marker_color=clr,
                opacity=0.88
            ))
        fig.update_layout(
            **base_layout(f"predict_proba() — First {n_show} Test Samples", h=380),
            barmode='group',
            xaxis_title="Test Sample",
            yaxis_title="Predicted Probability",
            yaxis=dict(range=[0,1], gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
            legend=dict(orientation='h', y=1.12, font=dict(color="#c8d6f0")),
            margin=dict(l=55, r=25, t=50, b=50),
            xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Confusion matrix
    left2, right2 = st.columns(2)

    with left2:
        cm = confusion_matrix(yite, ypred)
        fig2 = go.Figure(go.Heatmap(
            z=cm,
            x=[c.capitalize() for c in cls_names],
            y=[c.capitalize() for c in cls_names],
            colorscale=[[0, '#0d1929'], [0.5, '#1e3a7a'], [1, '#00d4ff']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=16, color='white'),
            showscale=False
        ))
        fig2.update_layout(
            **base_layout("Confusion Matrix", h=300),
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(l=70, r=20, t=50, b=60),
            xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
            yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with right2:
        # Per-class accuracy
        per_cls_acc = cm.diagonal() / cm.sum(axis=1)
        fig3 = go.Figure(go.Bar(
            x=[c.capitalize() for c in cls_names],
            y=per_cls_acc * 100,
            marker=dict(
                color=[ACCENT1, ACCENT2, ACCENT3],
                line=dict(color=PLOT_BG, width=1)
            ),
            text=[f"{v*100:.1f}%" for v in per_cls_acc],
            textposition='outside',
            textfont=dict(color='#e8f0ff', size=13)
        ))
        fig3.update_layout(
            **base_layout("Per-Class Accuracy (%)", h=300),
            yaxis=dict(range=[0, 115], gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
            margin=dict(l=50, r=20, t=50, b=50),
            xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR)
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # predict_proba table
    st.markdown('<div class="section-title" style="margin-top:8px;">predict_proba() <span>Output Table</span></div>', unsafe_allow_html=True)
    n_table = min(15, len(proba))
    df_proba = pd.DataFrame({
        "#": range(1, n_table+1),
        "True Label": [cls_names[yite[i]].capitalize() for i in range(n_table)],
        "Predicted":  [cls_names[ypred[i]].capitalize() for i in range(n_table)],
        "P(Setosa)":     proba[:n_table, 0].round(4),
        "P(Versicolor)": proba[:n_table, 1].round(4),
        "P(Virginica)":  proba[:n_table, 2].round(4),
        "Correct": ["✅" if yite[i]==ypred[i] else "❌" for i in range(n_table)]
    })
    st.dataframe(
        df_proba.style
            .background_gradient(cmap='Blues', subset=["P(Setosa)", "P(Versicolor)", "P(Virginica)"])
            .format({"P(Setosa)": "{:.4f}", "P(Versicolor)": "{:.4f}", "P(Virginica)": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f'<div class="info-box">✅ Each row sums to 1.0. OvR trains {len(cls_names)} independent binary classifiers then normalizes probabilities. Row 0 probability sum: <b>{proba[0].sum():.6f}</b></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:20px; border-top:1px solid #1a2e4a; margin-top:10px;">
  <span style="font-size:0.75rem; color:#3a5070; font-family:'Space Mono',monospace; letter-spacing:1px;">
    LOGISTIC REGRESSION LAB  ·  MACHINE LEARNING COURSE  ·  BSCS 6TH SEMESTER
  </span>
</div>
""", unsafe_allow_html=True)
