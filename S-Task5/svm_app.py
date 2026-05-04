import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SVM Kernel Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

*, html, body { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp { background: #04070e; }

section[data-testid="stSidebar"] {
    background: #070b14 !important;
    border-right: 1px solid #0c1a2e;
}
section[data-testid="stSidebar"] * { color: #6080a0 !important; }

.hero {
    background: linear-gradient(150deg, #05091a 0%, #0a1230 55%, #05091a 100%);
    border: 1px solid #0c1e38;
    border-radius: 20px;
    padding: 46px 54px;
    margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-100px; right:-80px;
    width:400px; height:400px;
    background: radial-gradient(circle, rgba(168,85,247,0.06) 0%, transparent 65%);
}
.hero::after {
    content:''; position:absolute; bottom:-70px; left:20%;
    width:300px; height:300px;
    background: radial-gradient(circle, rgba(59,130,246,0.05) 0%, transparent 65%);
}
.eyebrow {
    font-family:'Space Mono',monospace !important;
    font-size:0.63rem; color:#a855f7;
    letter-spacing:3px; text-transform:uppercase;
    margin-bottom:14px; display:block;
}
.hero-title {
    font-size:3.2rem; font-weight:800;
    color:#ffffff; line-height:0.95;
    margin:0 0 14px 0; letter-spacing:-0.5px;
}
.hero-title span { color:#a855f7; }
.hero-title em   { color:#3b82f6; font-style:normal; }
.hero-sub { color:#2a4060; font-size:0.88rem; max-width:540px; line-height:1.7; margin:0; }
.badge {
    display:inline-block; padding:4px 13px; border-radius:6px;
    font-size:0.65rem; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; margin:14px 3px 0 0;
    font-family:'Space Mono',monospace !important;
}
.b-purple{ background:rgba(168,85,247,0.1); border:1px solid rgba(168,85,247,0.25); color:#c084fc; }
.b-blue  { background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.25); color:#60a5fa; }
.b-green { background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); color:#34d399; }

.kpi {
    background:#070b14; border:1px solid #0c1a2e;
    border-radius:14px; padding:22px 18px; text-align:center;
    position:relative; overflow:hidden;
    transition:border-color 0.2s, transform 0.2s;
}
.kpi:hover { border-color:#162540; transform:translateY(-2px); }
.kpi::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:var(--c, #a855f7);
}
.kpi-v {
    font-family:'Space Mono',monospace !important;
    font-size:1.9rem; font-weight:700;
    color:var(--c, #a855f7); line-height:1; margin-bottom:6px;
}
.kpi-l { font-size:0.68rem; color:#1a3050; text-transform:uppercase; letter-spacing:1.4px; font-weight:600; }

.sec {
    font-size:1.45rem; font-weight:700; color:#d8e8ff;
    margin:8px 0 18px 0; padding-bottom:10px;
    border-bottom:1px solid #0c1a2e; letter-spacing:-0.2px;
}
.sec span { color:#a855f7; }

.box {
    background:#070b14; border:1px solid #0c1a2e;
    border-left:3px solid #a855f7; border-radius:0 10px 10px 0;
    padding:14px 18px; margin:12px 0;
    font-family:'Space Mono',monospace !important;
    font-size:0.78rem; color:#3a5a80; line-height:1.9;
}
.box b { color:#a0b8d8; }
.box.blue  { border-left-color:#3b82f6; }
.box.green { border-left-color:#10b981; }

.stButton>button {
    background:linear-gradient(135deg,#7c3aed,#3b82f6) !important;
    color:white !important; border:none !important;
    border-radius:8px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:700 !important; padding:10px 28px !important;
}
div[data-testid="stTabs"] button {
    color:#1a3050 !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:600 !important; font-size:0.87rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color:#a855f7 !important; border-bottom-color:#a855f7 !important;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#04070e; }
::-webkit-scrollbar-thumb { background:#0c1a2e; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
BG    = "#04070e"
PAPER = "#070b14"
GRID  = "#0c1825"
FONT  = "#2a4060"
PRP   = "#a855f7"
BLU   = "#3b82f6"
GRN   = "#10b981"
YEL   = "#f59e0b"
RED   = "#ef4444"
CYAN  = "#06b6d4"
CLRS  = [PRP, BLU, GRN, YEL, RED]

def bl(title="", h=400):
    return dict(
        title=dict(text=title, font=dict(color="#c8d6f0", size=13, family="Plus Jakarta Sans")),
        plot_bgcolor=BG, paper_bgcolor=PAPER,
        font=dict(color=FONT, family="Plus Jakarta Sans"),
        height=h,
        margin=dict(l=55, r=25, t=52, b=50),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span style="font-family:Space Mono;font-size:0.6rem;color:#1a3050;letter-spacing:2.5px;text-transform:uppercase;">⚙ CONTROLS</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    n_synth   = st.select_slider("Synthetic Samples", [200,300,400,600,800], value=400)
    noise_m   = st.slider("Moons Noise",   0.05, 0.40, 0.20, step=0.05)
    noise_c   = st.slider("Circles Noise", 0.05, 0.30, 0.15, step=0.05)
    svm_C     = st.select_slider("SVM C (Synthetic)", [0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)
    test_sz   = st.slider("BC Test Split", 0.15, 0.35, 0.20, step=0.05)
    bd_res    = st.select_slider("Boundary Resolution", [120, 180, 240], value=180)

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔁  Retrain", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span style="font-family:Space Mono;font-size:0.6rem;color:#1a3050;letter-spacing:2px;text-transform:uppercase;">Kernels</span>', unsafe_allow_html=True)
    for lbl in ['Linear','Poly deg=2','Poly deg=3','RBF']:
        st.markdown(f'<span class="badge b-purple">{lbl}</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="eyebrow">🔬 Machine Learning · Support Vector Machines</span>
  <div class="hero-title"><span>SVM</span> Kernel<br><em>Comparison</em> Lab</div>
  <p class="hero-sub">Linear · Polynomial · RBF kernels · GridSearchCV tuning · 5-classifier comparison on Breast Cancer</p>
  <div>
    <span class="badge b-purple">make_moons</span>
    <span class="badge b-blue">make_circles</span>
    <span class="badge b-green">Breast Cancer</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PIPELINE  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def run_pipeline(n, nm, nc, C_syn, ts, _res):
    np.random.seed(42)
    # ── Synthetic
    Xm, ym = make_moons(n_samples=n, noise=nm, random_state=42)
    Xc, yc = make_circles(n_samples=n, noise=nc, factor=0.5, random_state=42)
    sm, sc = StandardScaler(), StandardScaler()
    Xm_s = sm.fit_transform(Xm); Xc_s = sc.fit_transform(Xc)

    kernels = [
        ('Linear',       SVC(kernel='linear', C=C_syn, probability=True, random_state=42)),
        ('Poly (deg=2)', SVC(kernel='poly', degree=2, C=C_syn, gamma='scale', coef0=1, probability=True, random_state=42)),
        ('Poly (deg=3)', SVC(kernel='poly', degree=3, C=C_syn, gamma='scale', coef0=1, probability=True, random_state=42)),
        ('RBF',          SVC(kernel='rbf',  C=C_syn, gamma='scale', probability=True, random_state=42)),
    ]

    synth_results = {}
    for name, svm in kernels:
        row = {}
        for ds_label, X, y in [('moons', Xm_s, ym), ('circles', Xc_s, yc)]:
            clf = SVC(**{k:v for k,v in svm.get_params().items()})
            clf.fit(X, y)
            row[ds_label] = {
                'clf': clf,
                'acc': accuracy_score(y, clf.predict(X)),
                'svs': len(clf.support_vectors_),
                'X': X, 'y': y,
            }
        synth_results[name] = row

    # ── Breast Cancer
    data = load_breast_cancer()
    X_bc, y_bc = data.data, data.target
    sbc = StandardScaler()
    X_bc_s = sbc.fit_transform(X_bc)
    Xtr, Xte, ytr, yte = train_test_split(X_bc_s, y_bc, test_size=ts, random_state=42, stratify=y_bc)

    # GridSearchCV
    param_grid = {'C': [0.01,0.1,1,10,100], 'gamma': [0.001,0.01,0.1,1,10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42),
                      param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    gs.fit(Xtr, ytr)
    bp = gs.best_params_
    scores_grid = gs.cv_results_['mean_test_score'].reshape(5, 5)

    # 5 classifiers
    classifiers = {
        f'SVM RBF': SVC(kernel='rbf', C=bp['C'], gamma=bp['gamma'], probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM Linear': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
    }
    clf_results = {}
    for cname, clf in classifiers.items():
        clf.fit(Xtr, ytr)
        yp   = clf.predict(Xte)
        yprb = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, yprb)
        clf_results[cname] = {
            'clf': clf, 'yp': yp, 'yprb': yprb,
            'acc':  accuracy_score(yte, yp),
            'f1':   f1_score(yte, yp, average='weighted'),
            'auc':  roc_auc_score(yte, yprb),
            'cm':   confusion_matrix(yte, yp),
            'fpr': fpr, 'tpr': tpr,
        }

    return dict(
        Xm_s=Xm_s, ym=ym, Xc_s=Xc_s, yc=yc,
        synth_results=synth_results,
        Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
        bp=bp, scores_grid=scores_grid,
        param_grid=param_grid,
        best_cv=gs.best_score_,
        clf_results=clf_results,
        bc_data=data,
    )

with st.spinner("Training SVMs + GridSearchCV (may take ~20s)..."):
    D = run_pipeline(n_synth, noise_m, noise_c, svm_C, test_sz, bd_res)

sr   = D['synth_results']
cr   = D['clf_results']
bp   = D['bp']
yte  = D['yte']

best_auc_model = max(cr, key=lambda k: cr[k]['auc'])
best_acc_model = max(cr, key=lambda k: cr[k]['acc'])

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
kpi_data = [
    (f"C={bp['C']}",                            "Best GridSearch C",   PRP),
    (f"γ={bp['gamma']}",                         "Best GridSearch γ",   BLU),
    (f"{D['best_cv']:.3f}",                      "Best CV Accuracy",    GRN),
    (f"{cr['SVM RBF']['auc']:.3f}",              "SVM RBF AUC-ROC",     PRP),
    (f"{cr[best_auc_model]['auc']:.3f}",         f"Best AUC ({best_auc_model[:6]})", YEL),
]
for col,(val,lbl,clr) in zip([c1,c2,c3,c4,c5], kpi_data):
    with col:
        st.markdown(f"""<div class="kpi" style="--c:{clr}">
          <div class="kpi-v">{val}</div>
          <div class="kpi-l">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌀  Synthetic Data",
    "🗺  Decision Boundaries",
    "🔍  GridSearchCV",
    "📊  Classifier Battle",
    "📈  ROC Curves",
    "🏆  Final Table",
])

# ══════════════════════════════════════════════
# TAB 1 — Synthetic Data
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec">Synthetic <span>Datasets</span> — make_moons & make_circles</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    for col, X, y, title, clr0 in [
        (left,  D['Xm_s'], D['ym'],  "make_moons",   CYAN),
        (right, D['Xc_s'], D['yc'],  "make_circles",  GRN),
    ]:
        with col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X[y==0,0], y=X[y==0,1], mode='markers',
                marker=dict(color=clr0, size=6, opacity=0.75),
                name='Class 0'
            ))
            fig.add_trace(go.Scatter(
                x=X[y==1,0], y=X[y==1,1], mode='markers',
                marker=dict(color=RED, size=6, opacity=0.75),
                name='Class 1'
            ))
            fig.update_layout(
                **bl(title, h=340),
                xaxis_title="Feature 1", yaxis_title="Feature 2",
                legend=dict(font=dict(color='#c8d6f0'), bgcolor=PAPER, orientation='h', y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Kernel accuracy summary
    st.markdown('<div class="sec" style="margin-top:4px;">Kernel <span>Accuracy Summary</span></div>', unsafe_allow_html=True)

    kn = list(sr.keys())
    rows_table = []
    for name in kn:
        rows_table.append({
            "Kernel": name,
            "Moons Acc":    round(sr[name]['moons']['acc'],4),
            "Moons SVs":    sr[name]['moons']['svs'],
            "Circles Acc":  round(sr[name]['circles']['acc'],4),
            "Circles SVs":  sr[name]['circles']['svs'],
        })
    df_syn = pd.DataFrame(rows_table)
    st.dataframe(
        df_syn.style.background_gradient(cmap='Purples', subset=['Moons Acc','Circles Acc']),
        use_container_width=True, hide_index=True
    )

    st.markdown(f"""<div class="box">
      <b>Why Linear kernel fails on Moons & Circles?</b><br>
      Both datasets are NOT linearly separable. Linear SVM draws a straight hyperplane — impossible to separate moon-shaped or ring-shaped clusters.<br><br>
      <b>RBF (Radial Basis Function)</b> maps data to infinite dimensions using: K(x,z) = exp(−γ||x−z||²)<br>
      This creates non-linear curved decision boundaries that perfectly separate complex shapes.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — Decision Boundaries
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">Decision <span>Boundaries</span> — All Kernels</div>', unsafe_allow_html=True)

    ds_choice = st.radio("Select Dataset", ["make_moons", "make_circles"], horizontal=True)
    ds_key    = 'moons' if 'moons' in ds_choice else 'circles'
    X_bd      = D['Xm_s'] if ds_key == 'moons' else D['Xc_s']
    y_bd      = D['ym']   if ds_key == 'moons' else D['yc']

    k_names = list(sr.keys())
    cols4   = st.columns(4)

    margin = 0.7
    x1min, x1max = X_bd[:,0].min()-margin, X_bd[:,0].max()+margin
    x2min, x2max = X_bd[:,1].min()-margin, X_bd[:,1].max()+margin
    res = bd_res
    xx, yy = np.meshgrid(np.linspace(x1min,x1max,res), np.linspace(x2min,x2max,res))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for col, kname in zip(cols4, k_names):
        clf    = sr[kname][ds_key]['clf']
        Z      = clf.predict(grid).reshape(xx.shape)
        acc    = sr[kname][ds_key]['acc']
        n_sv   = sr[kname][ds_key]['svs']
        sv_pts = clf.support_vectors_

        fig = go.Figure()
        # Boundary fill
        fig.add_trace(go.Heatmap(
            x=np.linspace(x1min,x1max,res),
            y=np.linspace(x2min,x2max,res),
            z=Z, colorscale=[[0,f'rgba(6,182,212,0.25)'],[1,f'rgba(239,68,68,0.25)']],
            showscale=False, hoverinfo='skip', opacity=1.0
        ))
        # Points
        for cls, clr in [(0,CYAN),(1,RED)]:
            mask = y_bd == cls
            fig.add_trace(go.Scatter(
                x=X_bd[mask,0], y=X_bd[mask,1], mode='markers',
                marker=dict(color=clr, size=5, opacity=0.8),
                name=f'C{cls}', showlegend=False
            ))
        # Support vectors
        fig.add_trace(go.Scatter(
            x=sv_pts[:,0], y=sv_pts[:,1], mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=9,
                        line=dict(color='white', width=1.2)),
            name='SVs', showlegend=False
        ))
        fig.update_layout(
            **bl(f"{kname}<br><sup>Acc={acc:.3f} | SVs={n_sv}</sup>", h=300),
            margin=dict(l=40,r=15,t=60,b=40),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        with col:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="box blue">
      <b>Support Vectors (white circles)</b> are the training points closest to the decision boundary. SVM maximizes the margin between classes using only these critical points.<br><br>
      K=1 (Linear): Many SVs — can't separate well → low confidence boundaries.<br>
      RBF: Fewer SVs needed — the kernel trick creates a perfect curved boundary.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — GridSearchCV
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec">GridSearchCV <span>Heatmap</span> — RBF SVM on Breast Cancer</div>', unsafe_allow_html=True)

    scores_grid = D['scores_grid']
    C_vals     = D['param_grid']['C']
    gamma_vals = D['param_grid']['gamma']

    left, right = st.columns([1.3, 1])

    with left:
        # Plotly heatmap
        annotations = []
        for i in range(len(C_vals)):
            for j in range(len(gamma_vals)):
                val = scores_grid[i,j]
                is_best = (C_vals[i]==bp['C'] and gamma_vals[j]==bp['gamma'])
                annotations.append(dict(
                    x=j, y=i,
                    text=f"<b>{'★' if is_best else ''}{val:.3f}</b>",
                    showarrow=False,
                    font=dict(size=12, color='white' if val<scores_grid.max()*0.9 else '#0d0020')
                ))

        fig = go.Figure(go.Heatmap(
            z=scores_grid,
            x=[str(g) for g in gamma_vals],
            y=[str(c) for c in C_vals],
            colorscale=[[0,PAPER],[0.3,'#2d1060'],[0.7,'#7c3aed'],[1,PRP]],
            showscale=True,
            colorbar=dict(title='CV Acc', tickfont=dict(color=FONT)),
            hovertemplate='C=%{y}<br>gamma=%{x}<br>CV Acc=%{z:.4f}<extra></extra>'
        ))
        fig.update_layout(
            **bl("GridSearch CV Accuracy — C (rows) vs gamma (cols)", h=400),
            annotations=annotations,
            xaxis_title="gamma",
            yaxis_title="C",
            margin=dict(l=70,r=20,t=55,b=65)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # Best params card
        st.markdown(f"""<div class="box" style="margin-top:0">
          <b>GridSearchCV Results</b><br><br>
          C values tested   : <b>{C_vals}</b><br>
          γ values tested   : <b>{gamma_vals}</b><br>
          CV strategy       : <b>StratifiedKFold(5)</b><br>
          Scoring metric    : <b>accuracy</b><br><br>
          ★ Best C          : <b style="color:#c084fc">{bp['C']}</b><br>
          ★ Best gamma      : <b style="color:#c084fc">{bp['gamma']}</b><br>
          ★ Best CV Acc     : <b style="color:#c084fc">{D['best_cv']:.4f}</b><br><br>
          <b>Grid size:</b> 5×5 = 25 combinations × 5-fold = 125 fits
        </div>""", unsafe_allow_html=True)

        # Score distribution
        flat_scores = scores_grid.flatten()
        fig2 = go.Figure(go.Histogram(
            x=flat_scores, nbinsx=12,
            marker=dict(color=PRP, opacity=0.8, line=dict(color=BG, width=1))
        ))
        fig2.add_vline(x=D['best_cv'], line=dict(color=YEL, dash='dot', width=2),
                       annotation_text=f"Best {D['best_cv']:.3f}",
                       annotation_font=dict(color=YEL, size=10))
        fig2.update_layout(
            **bl("Score Distribution Across Grid", h=250),
            xaxis_title="CV Accuracy",
            yaxis_title="Count",
            margin=dict(l=50,r=20,t=55,b=45)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="box">
      <b>How to read the heatmap:</b><br>
      Bright purple = high CV accuracy. Darker = lower accuracy.<br>
      Top-right corner (high C, high γ): tends to overfit — memorizes training noise.<br>
      Bottom-left (low C, low γ): too smooth — underfits.<br>
      The star ★ marks the optimal combination found by GridSearchCV.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — Classifier Battle
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec">5-Classifier <span>Battle</span> — Breast Cancer Dataset</div>', unsafe_allow_html=True)

    cnames = list(cr.keys())
    short  = ['SVM RBF','Log Reg','KNN (5)','Rand Forest','SVM Lin']
    metrics_list = ['Accuracy','F1 Score','AUC-ROC']
    scores_by_metric = {
        'Accuracy': [cr[n]['acc'] for n in cnames],
        'F1 Score': [cr[n]['f1']  for n in cnames],
        'AUC-ROC':  [cr[n]['auc'] for n in cnames],
    }

    # Grouped bar
    fig = go.Figure()
    x  = np.arange(len(metrics_list))
    for i,(cn,sn,clr) in enumerate(zip(cnames, short, CLRS)):
        vals = [cr[cn]['acc'], cr[cn]['f1'], cr[cn]['auc']]
        fig.add_trace(go.Bar(
            name=sn, x=metrics_list, y=vals,
            marker=dict(color=clr, opacity=0.88, line=dict(color=BG,width=0.8)),
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
            textfont=dict(color='#c8d6f0', size=10),
            width=0.15, offset=(i-2)*0.16
        ))
    fig.update_layout(
        **bl("Accuracy · F1 · AUC-ROC — All 5 Classifiers", h=430),
        barmode='overlay',
        yaxis=dict(range=[0.80,1.08], gridcolor=GRID),
        legend=dict(orientation='h', y=1.1, font=dict(color='#c8d6f0')),
        margin=dict(l=50,r=20,t=55,b=50)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Confusion matrices
    st.markdown('<div class="sec" style="margin-top:4px;">Confusion <span>Matrices</span></div>', unsafe_allow_html=True)
    cols5 = st.columns(5)
    bc_classes = ['Malignant','Benign']
    colorscales = [
        [[0,BG],[0.4,'#2d1060'],[1,PRP]],
        [[0,BG],[0.4,'#0a2048'],[1,BLU]],
        [[0,BG],[0.4,'#0a2a20'],[1,GRN]],
        [[0,BG],[0.4,'#2a1800'],[1,YEL]],
        [[0,BG],[0.4,'#2a0808'],[1,RED]],
    ]
    for col, cn, sn, cs in zip(cols5, cnames, short, colorscales):
        cm_val = cr[cn]['cm']
        fig_cm = go.Figure(go.Heatmap(
            z=cm_val,
            x=['Pred 0','Pred 1'], y=['True 0','True 1'],
            colorscale=cs, showscale=False,
            text=cm_val, texttemplate='<b>%{text}</b>',
            textfont=dict(size=14, color='white')
        ))
        fig_cm.update_layout(
            **bl(f"{sn}<br><sup>Acc={cr[cn]['acc']:.3f}</sup>", h=240),
            margin=dict(l=55,r=10,t=60,b=55),
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9))
        )
        with col:
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 5 — ROC Curves
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec">ROC <span>Curves</span> — All 5 Classifiers</div>', unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:
        fig = go.Figure()
        for (cn, sn, clr) in zip(cnames, short, CLRS):
            fpr = cr[cn]['fpr']; tpr = cr[cn]['tpr']
            auc = cr[cn]['auc']
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{sn} (AUC={auc:.3f})",
                line=dict(color=clr, width=2.5),
                hovertemplate=f"{sn}<br>FPR=%{{x:.3f}}<br>TPR=%{{y:.3f}}<extra></extra>"
            ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(color='white', dash='dot', width=1),
            name='Random (0.5)', hoverinfo='skip'
        ))
        fig.update_layout(
            **bl("ROC Curves — All Classifiers on Breast Cancer", h=460),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0,1], gridcolor=GRID),
            yaxis=dict(range=[0,1.02], gridcolor=GRID),
            legend=dict(orientation='v', x=0.55, y=0.15,
                        font=dict(color='#c8d6f0', size=10),
                        bgcolor=PAPER, bordercolor=GRID),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # AUC horizontal bars
        auc_vals = [cr[cn]['auc'] for cn in cnames]
        sorted_pairs = sorted(zip(short, auc_vals, CLRS), key=lambda x: x[1])
        fig2 = go.Figure(go.Bar(
            x=[v for _,v,_ in sorted_pairs],
            y=[n for n,_,_ in sorted_pairs],
            orientation='h',
            marker=dict(color=[c for _,_,c in sorted_pairs],
                        line=dict(color=BG,width=0.5)),
            text=[f'{v:.4f}' for _,v,_ in sorted_pairs],
            textposition='outside',
            textfont=dict(color='#c8d6f0', size=11)
        ))
        fig2.update_layout(
            **bl("AUC-ROC Ranking", h=280),
            xaxis=dict(range=[0.90, 1.01], gridcolor=GRID),
            margin=dict(l=100,r=60,t=55,b=40)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Radar
        radar_metrics = ['Accuracy','F1 Score','AUC-ROC']
        fig3 = go.Figure()
        for cn, sn, clr in zip(cnames, short, CLRS):
            vals = [cr[cn]['acc'], cr[cn]['f1'], cr[cn]['auc']]
            fig3.add_trace(go.Scatterpolar(
                r=vals+[vals[0]],
                theta=radar_metrics+[radar_metrics[0]],
                fill='toself',
                fillcolor=f'rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.08)',
                line=dict(color=clr, width=2),
                name=sn
            ))
        fig3.update_layout(
            polar=dict(
                bgcolor=BG,
                radialaxis=dict(visible=True, range=[0.85,1], gridcolor=GRID,
                                color=FONT, tickfont=dict(size=8)),
                angularaxis=dict(color='#c8d6f0', tickfont=dict(size=10))
            ),
            paper_bgcolor=PAPER, height=280,
            legend=dict(font=dict(color='#c8d6f0', size=9), bgcolor=PAPER,
                        orientation='h', y=-0.12),
            margin=dict(l=40,r=40,t=30,b=50),
            title=dict(text="Radar — All Metrics", font=dict(color='#c8d6f0', size=12))
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 6 — Final Table
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec">Final <span>Comparison</span> Table — All 5 Classifiers</div>', unsafe_allow_html=True)

    kernel_desc = {
        'SVM RBF':      f'RBF  C={bp["C"]}  γ={bp["gamma"]}',
        'Logistic Regression': 'L2 Reg  C=1.0',
        'KNN (K=5)':    'Euclidean  K=5',
        'Random Forest':'Gini  100 trees',
        'SVM Linear':   'Linear  C=1.0',
    }

    df_final = pd.DataFrame([{
        'Model':     sn,
        'Config':    kernel_desc.get(sn, '—'),
        'Dataset':   'Breast Cancer (569)',
        'Accuracy':  round(cr[cn]['acc'],4),
        'F1 Score':  round(cr[cn]['f1'],4),
        'AUC-ROC':   round(cr[cn]['auc'],4),
        'Best?':     '⭐' if cr[cn]['auc'] == max(cr[c]['auc'] for c in cr) else '',
    } for cn, sn in zip(cnames, short)])

    st.dataframe(
        df_final.style
            .format({'Accuracy':'{:.4f}','F1 Score':'{:.4f}','AUC-ROC':'{:.4f}'})
            .background_gradient(cmap='Purples', subset=['Accuracy','F1 Score','AUC-ROC']),
        use_container_width=True, hide_index=True
    )

    # Gauge row
    st.markdown("<br>", unsafe_allow_html=True)
    gauge_cols = st.columns(5)
    for col, cn, sn, clr in zip(gauge_cols, cnames, short, CLRS):
        auc_v = cr[cn]['auc']
        r,g,b = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=auc_v,
            number=dict(font=dict(color=clr, size=22, family='Space Mono'), valueformat='.3f'),
            title=dict(text=sn, font=dict(color='#c8d6f0', size=11)),
            gauge=dict(
                axis=dict(range=[0.85,1.0], tickfont=dict(color=FONT, size=8)),
                bar=dict(color=clr, thickness=0.3),
                bgcolor=GRID, bordercolor=GRID,
                steps=[
                    dict(range=[0.85,0.92], color=f'rgba({r},{g},{b},0.05)'),
                    dict(range=[0.92,0.97], color=f'rgba({r},{g},{b},0.10)'),
                    dict(range=[0.97,1.00], color=f'rgba({r},{g},{b},0.18)'),
                ],
            )
        ))
        fig_g.update_layout(
            paper_bgcolor=PAPER, plot_bgcolor=BG,
            height=200, margin=dict(l=20,r=20,t=40,b=10)
        )
        with col:
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="box green">
      <b>Summary Insights:</b><br>
      SVM RBF (tuned) — Often best on tabular data with proper scaling. GridSearchCV found C={bp['C']}, γ={bp['gamma']}.<br>
      Random Forest — Strong baseline, no scaling needed, handles non-linearity via tree ensembles.<br>
      Logistic Regression — Fast, interpretable, competitive on linearly-separable cancer data.<br>
      KNN — Simple but powerful. Sensitive to scale (StandardScaler applied).<br>
      SVM Linear — Fast to train, works well when data is roughly linearly separable.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px;border-top:1px solid #0c1a2e;margin-top:20px;">
  <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#0c1a2e;letter-spacing:2px;">
    SVM KERNEL COMPARISON LAB  ·  BREAST CANCER + SYNTHETIC  ·  ML COURSE  ·  BSCS 6TH SEMESTER
  </span>
</div>
""", unsafe_allow_html=True)
