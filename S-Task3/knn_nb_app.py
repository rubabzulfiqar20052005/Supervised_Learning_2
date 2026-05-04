import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KNN vs Naive Bayes",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;700&display=swap');

*, html, body { font-family: 'Outfit', sans-serif !important; }
.stApp { background: #060a10; }

section[data-testid="stSidebar"] {
    background: #090e18 !important;
    border-right: 1px solid #0f1e35;
}
section[data-testid="stSidebar"] * { color: #7090b0 !important; }

.hero {
    background: linear-gradient(145deg, #07111f 0%, #0c1c38 50%, #07111f 100%);
    border: 1px solid #0f2040;
    border-radius: 20px;
    padding: 46px 54px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';position:absolute;top:-120px;right:-60px;
    width:420px;height:420px;
    background: radial-gradient(circle, rgba(6,182,212,0.05) 0%, transparent 65%);
}
.hero::after {
    content:'';position:absolute;bottom:-80px;left:25%;
    width:300px;height:300px;
    background: radial-gradient(circle, rgba(139,92,246,0.04) 0%, transparent 65%);
}
.eyebrow {
    font-family:'Fira Code',monospace !important;
    font-size:0.65rem;color:#06b6d4;
    letter-spacing:3px;text-transform:uppercase;
    margin-bottom:14px;display:block;
}
.hero-title {
    font-size:3.2rem;font-weight:800;
    color:#ffffff;line-height:1.0;
    margin:0 0 12px 0;letter-spacing:-0.5px;
}
.hero-title span { color:#06b6d4; }
.hero-title em   { color:#8b5cf6;font-style:normal; }
.hero-sub { color:#3d5575;font-size:0.9rem;max-width:520px;line-height:1.65;margin:0; }

.badge {
    display:inline-block;padding:4px 13px;border-radius:6px;
    font-size:0.67rem;font-weight:700;letter-spacing:1.5px;
    text-transform:uppercase;margin:14px 4px 0 0;
    font-family:'Fira Code',monospace !important;
}
.badge-cyan  { background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.25);color:#06b6d4; }
.badge-purple{ background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);color:#a78bfa; }
.badge-green { background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);color:#34d399; }

.card {
    background:#090e18;border:1px solid #0f1e35;
    border-radius:14px;padding:22px 20px;text-align:center;
    position:relative;overflow:hidden;
    transition:border-color 0.2s,transform 0.2s;
}
.card:hover { border-color:#1a3050;transform:translateY(-2px); }
.card::before {
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:var(--c,#06b6d4);
}
.card-val {
    font-family:'Fira Code',monospace !important;
    font-size:2.1rem;font-weight:700;
    color:var(--c,#06b6d4);line-height:1;margin-bottom:6px;
}
.card-lbl { font-size:0.7rem;color:#1e3a55;text-transform:uppercase;letter-spacing:1.4px;font-weight:600; }

.sec-title {
    font-size:1.5rem;font-weight:700;color:#dde8f8;
    margin:8px 0 20px 0;padding-bottom:10px;
    border-bottom:1px solid #0f1e35;letter-spacing:-0.2px;
}
.sec-title span { color:#06b6d4; }

.info {
    background:#090e18;border:1px solid #0f1e35;
    border-left:3px solid #06b6d4;border-radius:0 10px 10px 0;
    padding:15px 20px;margin:14px 0;
    font-family:'Fira Code',monospace !important;
    font-size:0.8rem;color:#5070a0;line-height:1.9;
}
.info b { color:#b0c8e8; }
.info.purple { border-left-color:#8b5cf6; }
.info.green  { border-left-color:#10b981; }

.stButton > button {
    background:linear-gradient(135deg,#0891b2,#7c3aed) !important;
    color:white !important;border:none !important;
    border-radius:8px !important;font-family:'Outfit',sans-serif !important;
    font-weight:700 !important;padding:10px 28px !important;
}
div[data-testid="stTabs"] button {
    color:#1e3a55 !important;font-family:'Outfit',sans-serif !important;
    font-weight:600 !important;font-size:0.88rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color:#06b6d4 !important;border-bottom-color:#06b6d4 !important;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#060a10; }
::-webkit-scrollbar-thumb { background:#0f1e35;border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────
BG    = "#060a10"
PAPER = "#090e18"
GRID  = "#0d1e30"
FONT  = "#3d5575"
CYAN  = "#06b6d4"
PRP   = "#8b5cf6"
GRN   = "#10b981"
YEL   = "#f59e0b"
RED   = "#ef4444"
ORG   = "#f97316"

PALETTE = [RED,ORG,YEL,GRN,CYAN,"#3b82f6",PRP,"#ec4899","#14b8a6",YEL]

def base_layout(title="", h=420):
    return dict(
        title=dict(text=title, font=dict(color="#c8d6f0", size=13, family="Outfit")),
        plot_bgcolor=BG, paper_bgcolor=PAPER,
        font=dict(color=FONT, family="Outfit"),
        height=h,
        margin=dict(l=55, r=25, t=50, b=50),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span style="font-family:Fira Code;font-size:0.62rem;color:#1e3a55;letter-spacing:2.5px;text-transform:uppercase;">⚙ CONTROLS</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    test_sz       = st.slider("Test Split", 0.15, 0.40, 0.20, step=0.05)
    max_k         = st.slider("Max K to Evaluate", 10, 30, 20)
    pca_mesh_res  = st.select_slider("Boundary Resolution", [100, 150, 200, 250], value=150)
    tfidf_feats   = st.select_slider("TF-IDF Max Features", [1000,2000,3000,5000], value=3000)
    docs_per_cat  = st.slider("Docs per Category", 100, 500, 300, step=50)

    st.markdown("<br>", unsafe_allow_html=True)
    retrain_btn = st.button("🔁  Retrain All Models", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span style="font-family:Fira Code;font-size:0.62rem;color:#1e3a55;letter-spacing:2px;text-transform:uppercase;">Algorithms</span>', unsafe_allow_html=True)
    st.markdown('<br><span class="badge badge-cyan">KNN</span><span class="badge badge-purple">GaussianNB</span><span class="badge badge-green">MultinomialNB</span>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<span style="font-family:Fira Code;font-size:0.62rem;color:#1e3a55;letter-spacing:2px;text-transform:uppercase;">Datasets</span>', unsafe_allow_html=True)
    st.markdown('<br><span class="badge badge-cyan">Digits (sklearn)</span><span class="badge badge-green">Synthetic News</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="eyebrow">🤖 Supervised Learning · Classification Comparison</span>
  <div class="hero-title"><span>KNN</span> vs <em>Naive Bayes</em></div>
  <p class="hero-sub">Elbow curve · Decision boundaries via PCA · GaussianNB vs KNN · MultinomialNB for text classification</p>
  <div>
    <span class="badge badge-cyan">Digits Dataset</span>
    <span class="badge badge-purple">K=1 to 20</span>
    <span class="badge badge-green">TF-IDF Text</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BUILD TEXT CORPUS
# ─────────────────────────────────────────────
@st.cache_data
def build_text_corpus(n_per_cat):
    np.random.seed(42)
    vocab_map = {
        'sports':     ['goal','team','player','game','win','score','match','league','coach','ball',
                       'season','championship','tournament','athlete','stadium','referee','penalty',
                       'transfer','manager','contract','trophy','fan','pitch','defender','striker'],
        'politics':   ['government','election','president','policy','vote','senate','congress','law',
                       'minister','campaign','party','debate','reform','democracy','bill','parliament',
                       'diplomat','treaty','constitution','tax','opposition','rally','candidate','ballot'],
        'technology': ['software','computer','internet','algorithm','data','network','code','server',
                       'cloud','artificial','machine','digital','system','device','platform','startup',
                       'cybersecurity','blockchain','quantum','robotics','processor','database','API','framework'],
        'science':    ['research','study','experiment','discovery','theory','molecule','particle','space',
                       'climate','biology','physics','chemistry','laboratory','scientist','evolution',
                       'genome','asteroid','gravity','neuron','ecosystem','fossil','telescope','microscope'],
        'business':   ['market','company','stock','economy','trade','profit','invest','finance','revenue',
                       'startup','corporate','CEO','industry','growth','merger','acquisition','dividend',
                       'bankruptcy','tariff','inflation','shareholder','quarterly','fiscal','supply'],
        'health':     ['disease','treatment','medicine','patient','hospital','doctor','therapy','vaccine',
                       'symptom','diet','exercise','mental','clinical','surgery','drug','nutrition',
                       'pandemic','antibiotic','diagnosis','wellbeing','immune','chronic','medical'],
    }
    common = ['the','is','was','are','has','have','this','that','with','for','and','but',
              'or','in','on','at','to','of','a','an','it','its','from','about']
    docs, labels, cat_names = [], [], list(vocab_map.keys())
    for i,(cat,words) in enumerate(vocab_map.items()):
        for _ in range(n_per_cat):
            w = np.random.choice(words, size=54, replace=True).tolist()
            w += np.random.choice(common, size=26, replace=True).tolist()
            np.random.shuffle(w)
            docs.append(' '.join(w)); labels.append(i)
    return np.array(docs), np.array(labels), cat_names

# ─────────────────────────────────────────────
# FULL PIPELINE (cached)
# ─────────────────────────────────────────────
@st.cache_data
def run_all(ts, mk, pr, tf, dpc):
    # ── Digits
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=ts, random_state=42, stratify=y)
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.20, random_state=42, stratify=ytr)

    # KNN sweep
    k_vals, val_accs, train_accs = [], [], []
    for k in range(1, mk+1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Xtr2, ytr2)
        val_accs.append(accuracy_score(yval, knn.predict(Xval)))
        train_accs.append(accuracy_score(ytr2, knn.predict(Xtr2)))
        k_vals.append(k)

    best_k   = k_vals[int(np.argmax(val_accs))]
    best_acc = max(val_accs)

    # Best KNN full test
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(Xtr, ytr)
    yp_knn = best_knn.predict(Xte)

    # GaussianNB
    gnb = GaussianNB(); gnb.fit(Xtr, ytr)
    yp_gnb = gnb.predict(Xte)

    # PCA for boundary
    pca = PCA(n_components=2, random_state=42)
    Xtr_pca = pca.fit_transform(Xtr); Xte_pca = pca.transform(Xte)
    Xall_pca = pca.transform(Xs)
    var_explained = pca.explained_variance_ratio_

    # Text
    text_docs, text_labels, cat_names = build_text_corpus(dpc)
    tfidf = TfidfVectorizer(max_features=tf, ngram_range=(1,2), sublinear_tf=True)
    Xt = tfidf.fit_transform(text_docs)
    Xttr, Xtte, yttr, ytte = train_test_split(Xt, text_labels, test_size=ts, random_state=42, stratify=text_labels)
    mnb = MultinomialNB(alpha=0.1); mnb.fit(Xttr, yttr)
    yp_mnb = mnb.predict(Xtte)

    return dict(
        X=X, y=y, Xs=Xs, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
        k_vals=k_vals, val_accs=val_accs, train_accs=train_accs,
        best_k=best_k, best_acc=best_acc,
        yp_knn=yp_knn, yp_gnb=yp_gnb,
        Xtr_pca=Xtr_pca, Xte_pca=Xte_pca, Xall_pca=Xall_pca,
        var_explained=var_explained, ytr_full=ytr, Xtr2=Xtr2, ytr2=ytr2,
        text_docs=text_docs, text_labels=text_labels, cat_names=cat_names,
        yp_mnb=yp_mnb, ytte=ytte,
        digits=load_digits(),
    )

with st.spinner("Training KNN × 20 + GaussianNB + MultinomialNB..."):
    D = run_all(test_sz, max_k, pca_mesh_res, tfidf_feats, docs_per_cat)

# ── convenience unpacking
k_vals      = D['k_vals'];     val_accs   = D['val_accs']
train_accs  = D['train_accs']; best_k     = D['best_k']
best_acc    = D['best_acc'];   yte        = D['yte']
yp_knn      = D['yp_knn'];     yp_gnb     = D['yp_gnb']
yp_mnb      = D['yp_mnb'];     ytte       = D['ytte']
cat_names   = D['cat_names'];  digits_obj = D['digits']

knn_acc  = accuracy_score(yte, yp_knn);  knn_prec = precision_score(yte, yp_knn, average='weighted')
knn_rec  = recall_score(yte, yp_knn, average='weighted'); knn_f1 = f1_score(yte, yp_knn, average='weighted')
gnb_acc  = accuracy_score(yte, yp_gnb);  gnb_prec = precision_score(yte, yp_gnb, average='weighted')
gnb_rec  = recall_score(yte, yp_gnb, average='weighted'); gnb_f1 = f1_score(yte, yp_gnb, average='weighted')
mnb_acc  = accuracy_score(ytte, yp_mnb); mnb_prec = precision_score(ytte, yp_mnb, average='weighted')
mnb_rec  = recall_score(ytte, yp_mnb, average='weighted'); mnb_f1 = f1_score(ytte, yp_mnb, average='weighted')

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
kpi_data = [
    (f"K = {best_k}",           "Best K (Elbow)",           CYAN,  CYAN),
    (f"{best_acc:.3f}",         "Best Val Accuracy",         CYAN,  CYAN),
    (f"{knn_acc:.3f}",          f"KNN Test Acc (K={best_k})",PRP,   PRP),
    (f"{gnb_acc:.3f}",          "GaussianNB Test Acc",       GRN,   GRN),
    (f"{mnb_acc:.3f}",          "MultinomialNB Text Acc",    YEL,   YEL),
]
for col,(val,lbl,clr,_) in zip([c1,c2,c3,c4,c5], kpi_data):
    with col:
        st.markdown(f"""<div class="card" style="--c:{clr}">
          <div class="card-val">{val}</div>
          <div class="card-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Dataset",
    "📈  Elbow Curve",
    "🗺  Decision Boundary",
    "🧮  GaussianNB",
    "📝  Text (MNB)",
    "🏆  Final Comparison"
])

# ══════════════════════════════════════════════
# TAB 1 — Dataset
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">Digits <span>Dataset</span> Overview</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.5])

    with left:
        counts = np.bincount(D['y'])
        fig = go.Figure(go.Bar(
            x=[str(i) for i in range(10)],
            y=counts,
            marker=dict(color=PALETTE, line=dict(color=BG, width=1)),
            text=counts, textposition='outside',
            textfont=dict(color='#c8d6f0', size=11)
        ))
        fig.update_layout(
            **base_layout("Samples per Digit Class", h=320),
            xaxis_title="Digit", yaxis_title="Count",
            yaxis=dict(gridcolor=GRID, range=[0, max(counts)*1.18]),
            margin=dict(l=50, r=20, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown(f"""<div class="info">
          <b>Digits Dataset (sklearn)</b><br><br>
          Samples    : <b>1797</b><br>
          Features   : <b>64</b>  (8×8 pixel images flattened)<br>
          Classes    : <b>10</b>  (digits 0–9)<br>
          Balanced   : <b>Yes</b>  (~180 per class)<br><br>
          Preprocessing applied: <b>StandardScaler</b><br>
          Required for KNN (Euclidean distance sensitive to scale)
        </div>""", unsafe_allow_html=True)

        # PCA variance pie
        var = D['var_explained']
        others = 1.0 - sum(var)
        fig2 = go.Figure(go.Pie(
            labels=["PC1", "PC2", "Remaining 62 PCs"],
            values=[var[0], var[1], others],
            hole=0.55,
            marker=dict(colors=[CYAN, PRP, GRID], line=dict(color=BG, width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color="#c8d6f0"),
        ))
        fig2.update_layout(
            **base_layout(f"PCA Variance Explained", h=250),
            showlegend=False,
            annotations=[dict(
                text=f"{(var[0]+var[1])*100:.1f}%<br><span style='font-size:10px'>2-PC Total</span>",
                x=0.5,y=0.5,showarrow=False,font=dict(size=14,color='#c8d6f0')
            )],
            margin=dict(l=30,r=30,t=50,b=20)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # PCA scatter of all data
    st.markdown('<div class="sec-title" style="margin-top:4px;">PCA <span>2D Projection</span> of All Samples</div>', unsafe_allow_html=True)
    Xall_pca = D['Xall_pca']
    fig3 = go.Figure()
    for digit in range(10):
        mask = D['y'] == digit
        fig3.add_trace(go.Scatter(
            x=Xall_pca[mask,0], y=Xall_pca[mask,1],
            mode='markers',
            marker=dict(color=PALETTE[digit], size=4, opacity=0.6),
            name=f"Digit {digit}"
        ))
    fig3.update_layout(
        **base_layout("All 1797 samples projected onto PCA Component 1 & 2", h=400),
        xaxis_title="PC1", yaxis_title="PC2",
        legend=dict(orientation='h', y=-0.12, font=dict(color='#c8d6f0', size=10),
                    facecolor=PAPER)
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 2 — Elbow Curve
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">KNN <span>Elbow Curve</span> — Validation Accuracy vs K</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_vals, y=train_accs,
        mode='lines+markers', name='Training Accuracy',
        line=dict(color=PRP, width=2, dash='dash'),
        marker=dict(size=5),
        hovertemplate="K=%{x}<br>Train Acc=%{y:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=k_vals, y=val_accs,
        mode='lines+markers', name='Validation Accuracy',
        line=dict(color=CYAN, width=3),
        marker=dict(size=7, symbol='circle'),
        hovertemplate="K=%{x}<br>Val Acc=%{y:.4f}<extra></extra>",
        fill='tonexty', fillcolor='rgba(6,182,212,0.04)'
    ))
    fig.add_trace(go.Scatter(
        x=[best_k], y=[best_acc],
        mode='markers+text',
        marker=dict(color=YEL, size=16, symbol='star',
                    line=dict(color='white', width=1.5)),
        text=[f"  Best K={best_k}"], textposition='middle right',
        textfont=dict(color=YEL, size=12),
        name=f"Best K={best_k}"
    ))
    fig.add_vline(x=best_k, line=dict(color=YEL, dash='dot', width=1.5))
    fig.update_layout(
        **base_layout(f"Elbow Curve: K=1 to {max_k}  |  Best K={best_k} (Val Acc={best_acc:.4f})", h=450),
        xaxis_title="K (Number of Neighbors)",
        yaxis_title="Accuracy",
        xaxis=dict(tickvals=k_vals, gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        legend=dict(orientation='h', y=1.12, font=dict(color='#c8d6f0')),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Table
    df_k = pd.DataFrame({
        "K": k_vals,
        "Val Accuracy":   [round(v,4) for v in val_accs],
        "Train Accuracy": [round(v,4) for v in train_accs],
        "Overfit Gap":    [round(t-v,4) for t,v in zip(train_accs,val_accs)],
        "Status": ["⭐ BEST" if k==best_k else "" for k in k_vals]
    })
    st.dataframe(
        df_k.style.background_gradient(cmap='Blues', subset=['Val Accuracy']),
        use_container_width=True, hide_index=True
    )

    st.markdown(f"""<div class="info">
      <b>Why KNN has high training accuracy at K=1?</b><br>
      K=1 memorizes every training sample — distance to itself is 0, so it always predicts the correct class. This is overfitting. As K increases, the model generalizes better (lower train acc, higher val acc). The elbow at K=<b>{best_k}</b> shows the sweet spot.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — Decision Boundary
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">Decision <span>Boundaries</span> — PCA 2D (K=1, K=5, K=15)</div>', unsafe_allow_html=True)

    Xtr_pca = D['Xtr_pca']; ytr_pca = D['ytr']
    Xall_pca = D['Xall_pca']

    margin = 1.5
    x_min = Xall_pca[:,0].min()-margin; x_max = Xall_pca[:,0].max()+margin
    y_min_b = Xall_pca[:,1].min()-margin; y_max_b = Xall_pca[:,1].max()+margin

    res = pca_mesh_res
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, res),
        np.linspace(y_min_b, y_max_b, res)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    k_show = [1, 5, 15]
    cols_bound = st.columns(3)

    for col, k in zip(cols_bound, k_show):
        knn_2d = KNeighborsClassifier(n_neighbors=k)
        knn_2d.fit(Xtr_pca, ytr_pca)
        Z = knn_2d.predict(grid).reshape(xx.shape)
        acc_2d = accuracy_score(D['ytr'], knn_2d.predict(Xtr_pca))

        fig = go.Figure()
        # Boundary fill using contourf-style heatmap
        fig.add_trace(go.Heatmap(
            x=np.linspace(x_min, x_max, res),
            y=np.linspace(y_min_b, y_max_b, res),
            z=Z,
            colorscale=[[i/9, PALETTE[i]] for i in range(10)],
            opacity=0.22, showscale=False, hoverinfo='skip'
        ))
        for digit in range(10):
            mask = ytr_pca == digit
            fig.add_trace(go.Scatter(
                x=Xtr_pca[mask,0], y=Xtr_pca[mask,1],
                mode='markers',
                marker=dict(color=PALETTE[digit], size=4, opacity=0.75,
                            line=dict(color='rgba(255,255,255,0.2)', width=0.3)),
                name=str(digit), showlegend=(k==1)
            ))
        fig.update_layout(
            **base_layout(f"K = {k}  |  Train Acc: {acc_2d:.3f}", h=340),
            xaxis_title="PC1", yaxis_title="PC2",
            margin=dict(l=50,r=15,t=50,b=40),
            showlegend=False
        )
        with col:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Legend
    legend_html = " ".join([
        f'<span style="background:{PALETTE[i]};padding:3px 10px;border-radius:4px;'
        f'font-family:Fira Code;font-size:0.7rem;color:black;margin:2px;display:inline-block;">'
        f'Digit {i}</span>' for i in range(10)
    ])
    st.markdown(f'<div style="margin:12px 0;text-align:center;">{legend_html}</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info purple">
      <b>PCA Decision Boundary Insight:</b><br>
      K=1 → very jagged irregular boundaries (high variance, overfitting in 2D)<br>
      K=5 → smoother boundaries, good bias-variance balance<br>
      K=15 → very smooth boundaries (higher bias, may underfit)<br>
      Note: These use only 2 PCA features, so accuracy is lower than full 64-feature model.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — GaussianNB
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">Gaussian <span>Naive Bayes</span> — Digits Classification</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        # Confusion matrix
        cm = confusion_matrix(yte, yp_gnb)
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=[str(i) for i in range(10)],
            y=[str(i) for i in range(10)],
            colorscale=[[0,BG],[0.4,'#1a1060'],[1,PRP]],
            showscale=False,
            text=cm, texttemplate="<b>%{text}</b>",
            textfont=dict(size=11, color='white')
        ))
        fig.update_layout(
            **base_layout("GaussianNB Confusion Matrix", h=380),
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(l=55,r=20,t=55,b=60)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # Per-class F1
        cr_gnb = classification_report(yte, yp_gnb, output_dict=True)
        per_f1 = [cr_gnb[str(i)]['f1-score'] for i in range(10)]
        fig2 = go.Figure(go.Bar(
            x=[str(i) for i in range(10)],
            y=per_f1,
            marker=dict(color=PALETTE, line=dict(color=BG, width=1)),
            text=[f"{v:.2f}" for v in per_f1],
            textposition='outside',
            textfont=dict(color='#c8d6f0', size=10)
        ))
        fig2.update_layout(
            **base_layout("Per-Class F1 Score (GaussianNB)", h=380),
            xaxis_title="Digit Class",
            yaxis=dict(range=[0,1.15], gridcolor=GRID),
            margin=dict(l=50,r=20,t=55,b=55)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # KNN vs GNB comparison
    st.markdown('<div class="sec-title" style="margin-top:4px;">KNN vs <span>GaussianNB</span> — Side by Side</div>', unsafe_allow_html=True)

    metric_names = ['Accuracy','Precision','Recall','F1 Score']
    knn_scores = [knn_acc,knn_prec,knn_rec,knn_f1]
    gnb_scores = [gnb_acc,gnb_prec,gnb_rec,gnb_f1]
    x = np.arange(len(metric_names))
    w = 0.35

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=[m for m in metric_names], y=knn_scores,
                          name=f"KNN (K={best_k})",
                          marker_color=CYAN, opacity=0.9,
                          text=[f"{v:.3f}" for v in knn_scores],
                          textposition='outside',
                          textfont=dict(color='#c8d6f0', size=11),
                          width=0.3, offset=-0.17))
    fig3.add_trace(go.Bar(x=[m for m in metric_names], y=gnb_scores,
                          name="GaussianNB",
                          marker_color=PRP, opacity=0.9,
                          text=[f"{v:.3f}" for v in gnb_scores],
                          textposition='outside',
                          textfont=dict(color='#c8d6f0', size=11),
                          width=0.3, offset=0.17))
    fig3.update_layout(
        **base_layout("KNN vs GaussianNB — All Metrics", h=360),
        barmode='overlay',
        yaxis=dict(range=[0,1.18], gridcolor=GRID),
        legend=dict(orientation='h', y=1.12, font=dict(color='#c8d6f0')),
        margin=dict(l=50,r=20,t=55,b=50)
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="info purple">
      <b>GaussianNB Assumption:</b> Each feature follows a Gaussian (normal) distribution per class. It calculates P(class|features) using Bayes theorem: P(class|x) ∝ P(x|class) × P(class).<br><br>
      <b>Why KNN usually outperforms GNB on Digits?</b> Digit pixel features are not Gaussian — they are sparse, binary-like values. KNN makes no distributional assumptions, just finds nearest neighbors.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — Text / MultinomialNB
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">Multinomial <span>Naive Bayes</span> — Text Classification</div>', unsafe_allow_html=True)

    text_docs = D['text_docs']; text_labels = D['text_labels']
    total_docs = len(text_docs)
    counts_text = np.bincount(text_labels)

    left, right = st.columns([1, 1.3])

    with left:
        fig = go.Figure(go.Bar(
            x=[c.capitalize() for c in cat_names],
            y=counts_text,
            marker=dict(color=[CYAN,PRP,GRN,YEL,RED,ORG]),
            text=counts_text, textposition='outside',
            textfont=dict(color='#c8d6f0', size=11)
        ))
        fig.update_layout(
            **base_layout("Documents per Category", h=320),
            xaxis=dict(tickangle=-20),
            yaxis=dict(range=[0,max(counts_text)*1.2], gridcolor=GRID),
            margin=dict(l=50,r=20,t=55,b=70)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown(f"""<div class="info green">
          <b>Synthetic News Dataset</b><br><br>
          Categories   : <b>{len(cat_names)}</b> — {', '.join(cat_names)}<br>
          Total docs   : <b>{total_docs}</b>  ({docs_per_cat} per category)<br>
          TF-IDF feats : <b>{tfidf_feats}</b>  (unigrams + bigrams)<br><br>
          <b>Why MultinomialNB for text?</b><br>
          Text features are word counts / TF-IDF scores — non-negative integers/floats. MultinomialNB is designed for count-based features and works well with sparse TF-IDF matrices.
        </div>""", unsafe_allow_html=True)

    # Confusion matrix MNB
    cm_mnb = confusion_matrix(ytte, yp_mnb)
    cr_mnb = classification_report(ytte, yp_mnb, output_dict=True)

    left2, right2 = st.columns(2)
    with left2:
        fig2 = go.Figure(go.Heatmap(
            z=cm_mnb,
            x=[c.capitalize() for c in cat_names],
            y=[c.capitalize() for c in cat_names],
            colorscale=[[0,BG],[0.4,'#0a2a20'],[1,GRN]],
            showscale=False,
            text=cm_mnb, texttemplate="<b>%{text}</b>",
            textfont=dict(size=12,color='white')
        ))
        fig2.update_layout(
            **base_layout("MultinomialNB Confusion Matrix", h=360),
            xaxis=dict(tickangle=-25),
            margin=dict(l=80,r=20,t=55,b=85)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with right2:
        per_f1_mnb = [cr_mnb[str(i)]['f1-score'] for i in range(len(cat_names))]
        fig3 = go.Figure(go.Bar(
            x=[c.capitalize() for c in cat_names],
            y=per_f1_mnb,
            marker=dict(color=[CYAN,PRP,GRN,YEL,RED,ORG]),
            text=[f"{v:.3f}" for v in per_f1_mnb],
            textposition='outside',
            textfont=dict(color='#c8d6f0', size=11)
        ))
        fig3.update_layout(
            **base_layout("Per-Category F1 Score", h=360),
            xaxis=dict(tickangle=-20),
            yaxis=dict(range=[0,1.18],gridcolor=GRID),
            margin=dict(l=50,r=20,t=55,b=70)
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 6 — Final Comparison
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-title">Final <span>Comparison</span> Table — All Models</div>', unsafe_allow_html=True)

    # Summary dataframe
    df_final = pd.DataFrame({
        "Model":     [f"KNN (K={best_k})", "GaussianNB", "MultinomialNB (Text)"],
        "Dataset":   ["Digits (64 feat)","Digits (64 feat)","News TF-IDF 5k"],
        "Accuracy":  [knn_acc,  gnb_acc,  mnb_acc],
        "Precision": [knn_prec, gnb_prec, mnb_prec],
        "Recall":    [knn_rec,  gnb_rec,  mnb_rec],
        "F1 Score":  [knn_f1,   gnb_f1,   mnb_f1],
    })
    st.dataframe(
        df_final.style
            .format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1 Score":"{:.4f}"})
            .background_gradient(cmap='Blues', subset=['Accuracy','F1 Score']),
        use_container_width=True, hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar comparison
    metrics   = ['Accuracy','Precision','Recall','F1 Score']
    m_knn     = [knn_acc,  knn_prec,  knn_rec,  knn_f1]
    m_gnb     = [gnb_acc,  gnb_prec,  gnb_rec,  gnb_f1]
    m_mnb     = [mnb_acc,  mnb_prec,  mnb_rec,  mnb_f1]
    model_clrs = [CYAN, PRP, GRN]

    fig = go.Figure()
    for name, scores, clr in zip(
        [f"KNN (K={best_k})", "GaussianNB", "MultinomialNB"],
        [m_knn, m_gnb, m_mnb], model_clrs
    ):
        fig.add_trace(go.Bar(
            name=name, x=metrics, y=scores,
            marker=dict(color=clr, opacity=0.88, line=dict(color=BG,width=1)),
            text=[f"{v:.3f}" for v in scores],
            textposition='outside', textfont=dict(color='#c8d6f0', size=11)
        ))
    fig.update_layout(
        **base_layout("KNN vs GaussianNB vs MultinomialNB — Full Metric Comparison", h=430),
        barmode='group',
        yaxis=dict(range=[0,1.15], gridcolor=GRID),
        legend=dict(orientation='h', y=1.1, font=dict(color='#c8d6f0')),
        margin=dict(l=50,r=20,t=55,b=50)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Radar chart
    fig2 = go.Figure()
    for name, scores, clr in zip(
        [f"KNN (K={best_k})", "GaussianNB", "MultinomialNB"],
        [m_knn, m_gnb, m_mnb], model_clrs
    ):
        fig2.add_trace(go.Scatterpolar(
            r=scores+[scores[0]],
            theta=metrics+[metrics[0]],
            fill='toself', fillcolor=f'rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.1)',
            line=dict(color=clr,width=2.5),
            name=name
        ))
    fig2.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID,
                            color=FONT,tickfont=dict(size=9)),
            angularaxis=dict(color='#c8d6f0')
        ),
        paper_bgcolor=PAPER, height=380,
        legend=dict(font=dict(color='#c8d6f0'),bgcolor=PAPER,
                    orientation='h',y=-0.1),
        margin=dict(l=60,r=60,t=40,b=60),
        title=dict(text="Radar: All Models × All Metrics",
                   font=dict(color='#c8d6f0',size=13))
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="info green">
      <b>Key Takeaways:</b><br>
      KNN (K={best_k}) — Best overall on Digits. Non-parametric, no training assumptions. Slower at inference (computes distances to all training points).<br>
      GaussianNB — Fast, simple, but assumes Gaussian distribution per feature. Works well when this assumption holds (e.g., continuous numeric features).<br>
      MultinomialNB — Designed for discrete count data. Excellent for text classification with TF-IDF. Cannot be directly compared on Digits (different task).
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px;border-top:1px solid #0f1e35;margin-top:20px;">
  <span style="font-family:'Fira Code',monospace;font-size:0.63rem;color:#0f2035;letter-spacing:2px;">
    KNN vs NAIVE BAYES  ·  DIGITS + TEXT CLASSIFICATION  ·  ML COURSE  ·  BSCS 6TH SEMESTER
  </span>
</div>
""", unsafe_allow_html=True)
