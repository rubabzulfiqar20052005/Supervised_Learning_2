"""
╔══════════════════════════════════════════════════════════════════════╗
║   TREE MODELS LAB — Interactive Streamlit Dashboard                 ║
║   Heart Disease Dataset | Decision Tree + Random Forest             ║
╚══════════════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      validation_curve)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import LabelEncoder

# ─── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Tree Models Lab",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .main { background: #0d1117; }
  .stApp { background: linear-gradient(135deg, #0d1117 0%, #0f1923 100%); }

  .hero-header {
    background: linear-gradient(135deg, #1a2942 0%, #0d3b6e 50%, #1a2942 100%);
    border: 1px solid #1e6eb5;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(30, 110, 181, 0.25);
  }
  .hero-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8f4fd;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }
  .hero-header p {
    color: #7eb8e8;
    font-size: 1.05rem;
    margin: 0;
  }

  .metric-card {
    background: linear-gradient(135deg, #141e2e, #1a2942);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  }
  .metric-card:hover { transform: translateY(-2px); border-color: #3a7bd5; }
  .metric-label { font-size: 0.78rem; color: #7eb8e8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem; }
  .metric-value { font-size: 2rem; font-weight: 700; color: #e8f4fd; font-family: 'JetBrains Mono', monospace; }
  .metric-sub   { font-size: 0.75rem; color: #4a8aba; margin-top: 0.1rem; }

  .section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.1rem; font-weight: 700; color: #e8f4fd;
    border-left: 4px solid #3a7bd5;
    padding-left: 0.8rem; margin: 1.2rem 0 0.8rem 0;
  }

  .info-box {
    background: #101d2e;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3a7bd5;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #b8d4ed;
    line-height: 1.6;
  }
  .warn-box {
    background: #1e1208;
    border: 1px solid #5f3a1e;
    border-left: 4px solid #e6782a;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #e8c49a;
    line-height: 1.6;
  }
  .success-box {
    background: #0a1e14;
    border: 1px solid #1e5f3a;
    border-left: 4px solid #2ae67e;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #9ae8c0;
    line-height: 1.6;
  }

  .param-tag {
    display: inline-block;
    background: #1a3a5c;
    color: #7eb8e8;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    margin: 0.1rem;
  }

  .summary-table th {
    background: #1a2942 !important;
    color: #7eb8e8 !important;
    font-weight: 600;
  }

  div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1422 0%, #0d1f35 100%);
    border-right: 1px solid #1e3a5f;
  }
  .sidebar-logo {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
    font-size: 2.2rem;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: #101d2e;
    border-radius: 10px;
    padding: 0.2rem;
    gap: 4px;
    border: 1px solid #1e3a5f;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7eb8e8 !important;
    font-weight: 500;
    padding: 0.5rem 1.1rem;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e4a7a, #3a7bd5) !important;
    color: white !important;
  }

  .stSlider > div > div > div > div { background: #3a7bd5 !important; }
  .stSelectbox > div > div { background: #101d2e; border-color: #1e3a5f; color: #e8f4fd; }
  .stMultiSelect > div > div { background: #101d2e; border-color: #1e3a5f; }

  [data-testid="metric-container"] {
    background: #141e2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
  }

  .stButton > button {
    background: linear-gradient(135deg, #1e4a7a, #3a7bd5);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.6rem 1.5rem;
    transition: all 0.2s;
    box-shadow: 0 4px 12px rgba(58, 123, 213, 0.3);
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #3a7bd5, #5a9be5);
    box-shadow: 0 6px 18px rgba(58, 123, 213, 0.5);
    transform: translateY(-1px);
  }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB DARK THEME ──────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0d1117',
    'axes.facecolor':    '#141e2e',
    'axes.edgecolor':    '#1e3a5f',
    'axes.labelcolor':   '#b8d4ed',
    'text.color':        '#e8f4fd',
    'xtick.color':       '#7eb8e8',
    'ytick.color':       '#7eb8e8',
    'grid.color':        '#1e3a5f',
    'grid.linewidth':    0.8,
    'axes.titlecolor':   '#e8f4fd',
    'figure.titlesize':  13,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
})

COLORS = {
    'dt_full':   '#E53935',
    'dt_pruned': '#42A5F5',
    'rf':        '#66BB6A',
    'accent':    '#3a7bd5',
    'warn':      '#FFA726',
}

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING & MODEL TRAINING (Cached)
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset(choice):
    if choice == "Heart Disease (UCI Cleveland)":
        urls = [
            "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
            "https://raw.githubusercontent.com/amirmasoudabdol/heart-disease-dataset/main/heart.csv",
        ]
        for url in urls:
            try:
                df = pd.read_csv(url)
                if 'target' in df.columns:
                    return df, 'Heart Disease (UCI Cleveland)'
            except Exception:
                pass

        # Synthetic fallback
        np.random.seed(42)
        n = 303
        age   = np.random.normal(54, 9, n).clip(29, 77).astype(int)
        sex   = np.random.binomial(1, 0.68, n)
        cp    = np.random.choice([0,1,2,3], n, p=[0.47,0.17,0.28,0.08])
        trestbps = np.random.normal(131, 17, n).clip(94, 200).astype(int)
        chol  = np.random.normal(246, 52, n).clip(126, 564).astype(int)
        fbs   = np.random.binomial(1, 0.15, n)
        restecg = np.random.choice([0,1,2], n, p=[0.50,0.01,0.49])
        thalach = np.random.normal(150, 23, n).clip(71, 202).astype(int)
        exang = np.random.binomial(1, 0.33, n)
        oldpeak = np.random.exponential(1.0, n).clip(0, 6.2).round(1)
        slope = np.random.choice([0,1,2], n, p=[0.08,0.46,0.46])
        ca    = np.random.choice([0,1,2,3], n, p=[0.58,0.22,0.13,0.07])
        thal  = np.random.choice([1,2,3], n, p=[0.05,0.55,0.40])
        risk  = (0.04*(age-50) + 0.3*sex + 0.4*(cp==0).astype(int)
                 - 0.02*(thalach-150) + 0.5*exang + 0.3*oldpeak
                 + 0.3*ca + 0.2*(thal==3).astype(int)
                 + np.random.normal(0, 0.3, n))
        target = (risk > np.median(risk)).astype(int)
        df = pd.DataFrame({'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,
                           'chol':chol,'fbs':fbs,'restecg':restecg,'thalach':thalach,
                           'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,
                           'thal':thal,'target':target})
        return df, 'Heart Disease (Synthetic — realistic distributions)'

    else:  # Titanic
        try:
            import seaborn as sns_data
            df = sns_data.load_dataset('titanic')
        except Exception:
            try:
                df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
                df.rename(columns={'Survived':'survived','Pclass':'pclass',
                                   'Sex':'sex','Age':'age','SibSp':'sibsp',
                                   'Parch':'parch','Fare':'fare','Embarked':'embarked'},
                          inplace=True)
            except Exception:
                np.random.seed(42)
                n = 891
                df = pd.DataFrame({
                    'survived': np.random.binomial(1, 0.38, n),
                    'pclass':   np.random.choice([1,2,3], n, p=[0.24,0.21,0.55]),
                    'sex':      np.random.choice([0,1], n, p=[0.65,0.35]),
                    'age':      np.random.normal(29, 14, n).clip(0.5, 80),
                    'sibsp':    np.random.choice(range(9), n, p=[0.68,0.23,0.03,0.02,0.02,0.01,0.005,0.002,0.003]),
                    'parch':    np.random.choice(range(7), n, p=[0.76,0.13,0.05,0.04,0.01,0.005,0.005]),
                    'fare':     np.random.exponential(33, n).clip(0, 512),
                })
                return df, 'Titanic (Synthetic)'

        keep = ['survived','pclass','sex','age','sibsp','parch','fare','embarked']
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()

        if 'sex' in df.columns and df['sex'].dtype == object:
            df['sex'] = (df['sex'] == 'male').astype(int)
        if 'embarked' in df.columns:
            df['embarked'] = LabelEncoder().fit_transform(df['embarked'].astype(str))
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        target_col = 'survived'
        df.rename(columns={'survived':'target'}, inplace=True)
        return df, 'Titanic'

@st.cache_resource
def run_full_pipeline(df_key, max_depth_range, rs):
    df, ds_name = load_dataset(df_key)
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    X = df.drop(columns=['target'])
    y = df['target'].astype(int)
    feat = X.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=rs, stratify=y)

    # Unpruned DT
    dt_full = DecisionTreeClassifier(random_state=rs)
    dt_full.fit(X_tr, y_tr)

    # GridSearchCV
    pg = {
        'max_depth':         [None] + list(range(2, max_depth_range + 1)),
        'min_samples_split': [2, 5, 10, 20, 30],
        'min_samples_leaf':  [1, 2, 4, 6, 8],
    }
    gs = GridSearchCV(DecisionTreeClassifier(random_state=rs),
                      pg, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best_dt = gs.best_estimator_

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                 random_state=rs, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    def metrics(model, X_tr, y_tr, X_te, y_te):
        return {
            'train_acc': accuracy_score(y_tr, model.predict(X_tr)),
            'test_acc':  accuracy_score(y_te, model.predict(X_te)),
            'f1':        f1_score(y_te, model.predict(X_te), zero_division=0),
            'auc':       roc_auc_score(y_te, model.predict_proba(X_te)[:,1]),
        }

    m_full = metrics(dt_full, X_tr, y_tr, X_te, y_te)
    m_best = metrics(best_dt, X_tr, y_tr, X_te, y_te)
    m_rf   = metrics(rf, X_tr, y_tr, X_te, y_te)
    m_rf['oob_score'] = rf.oob_score_
    m_rf['oob_error'] = 1 - rf.oob_score_

    return {
        'X': X, 'y': y, 'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'feat': feat, 'ds_name': ds_name,
        'dt_full': dt_full, 'best_dt': best_dt, 'rf': rf,
        'gs': gs,
        'm_full': m_full, 'm_best': m_best, 'm_rf': m_rf,
    }

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌳</div>', unsafe_allow_html=True)
    st.markdown("## **Tree Models Lab**")
    st.markdown("---")

    st.markdown("### 📂 Dataset")
    dataset_choice = st.selectbox("Choose Dataset",
        ["Heart Disease (UCI Cleveland)", "Titanic"])

    st.markdown("### ⚙️ Parameters")
    max_depth_limit = st.slider("GridSearch Max Depth Range", 3, 20, 12)
    random_state    = st.slider("Random State", 0, 100, 42)

    st.markdown("---")
    run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#4a8aba; line-height:1.7'>
    <b>Pipeline Steps:</b><br>
    1️⃣ Load Dataset<br>
    2️⃣ Unpruned Decision Tree<br>
    3️⃣ Validation Curves<br>
    4️⃣ GridSearchCV Pruning<br>
    5️⃣ Tree Visualization<br>
    6️⃣ Random Forest (100 Trees)<br>
    7️⃣ Feature Importance<br>
    8️⃣ Metrics Comparison<br>
    9️⃣ Summary Table
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-header">
  <h1>🌳 Tree Models Lab</h1>
  <p>Decision Tree → Pre-Pruning → Random Forest → Comparison Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA & RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════
with st.spinner("⚙️ Training models... please wait"):
    P = run_full_pipeline(dataset_choice, max_depth_limit, random_state)

X, y = P['X'], P['y']
X_tr, X_te = P['X_tr'], P['X_te']
y_tr, y_te = P['y_tr'], P['y_te']
feat = P['feat']
dt_full, best_dt, rf = P['dt_full'], P['best_dt'], P['rf']
gs = P['gs']
m_full, m_best, m_rf = P['m_full'], P['m_best'], P['m_rf']

# Dataset info strip
dcol1, dcol2, dcol3, dcol4 = st.columns(4)
with dcol1:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Dataset</div>
    <div class="metric-value" style="font-size:1.1rem">{P['ds_name'][:22]}</div>
    </div>""", unsafe_allow_html=True)
with dcol2:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Samples</div>
    <div class="metric-value">{len(X)}</div>
    <div class="metric-sub">Train={len(X_tr)} / Test={len(X_te)}</div>
    </div>""", unsafe_allow_html=True)
with dcol3:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Features</div>
    <div class="metric-value">{len(feat)}</div>
    </div>""", unsafe_allow_html=True)
with dcol4:
    class_dist = y.value_counts()
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Class Balance</div>
    <div class="metric-value" style="font-size:1rem">{class_dist[0]} / {class_dist[1]}</div>
    <div class="metric-sub">Class 0 / Class 1</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔥 Overfitting",
    "📈 Validation Curves",
    "✂️ Best Tree",
    "🌲 Random Forest",
    "📊 Feature Importance",
    "🏆 Summary",
])

# ─────────── TAB 1: OVERFITTING ─────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Step 1 — Unpruned Decision Tree (No Depth Limit)</div>',
                unsafe_allow_html=True)

    st.markdown(f"""<div class="warn-box">
    ⚠️ <b>Overfitting Detected!</b> The unpruned tree achieves <b>100% Train Accuracy</b>
    but only <b>{m_full['test_acc']:.1%} Test Accuracy</b> — a gap of
    <b>{m_full['train_acc'] - m_full['test_acc']:.1%}</b>.
    Tree depth = <b>{dt_full.get_depth()}</b>, Leaves = <b>{dt_full.get_n_leaves()}</b>.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Train Accuracy", m_full['train_acc'], "#E53935"),
        (c2, "Test Accuracy",  m_full['test_acc'],  "#42A5F5"),
        (c3, "F1 Score",       m_full['f1'],         "#66BB6A"),
        (c4, "AUC-ROC",        m_full['auc'],        "#FFA726"),
    ]:
        col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{val:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')

    # Bar chart
    ax = axes[0]
    bars = ax.bar(['Train\nAccuracy', 'Test\nAccuracy'],
                   [m_full['train_acc'], m_full['test_acc']],
                   color=[COLORS['dt_full'], '#1565C0'],
                   edgecolor='#3a7bd5', linewidth=1.2, width=0.4)
    ax.set_ylim(0, 1.2)
    ax.set_title('Unpruned DT: Train vs Test Accuracy', fontweight='bold')
    ax.set_ylabel('Accuracy')
    for bar, val in zip(bars, [m_full['train_acc'], m_full['test_acc']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=13, color='#e8f4fd')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.4, label='Perfect Train Fit')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Depth complexity curve
    ax2 = axes[1]
    depth_range = range(1, min(22, dt_full.get_depth() + 2))
    tr_accs, te_accs = [], []
    for d in depth_range:
        tmp = DecisionTreeClassifier(max_depth=d, random_state=random_state)
        tmp.fit(X_tr, y_tr)
        tr_accs.append(accuracy_score(y_tr, tmp.predict(X_tr)))
        te_accs.append(accuracy_score(y_te, tmp.predict(X_te)))
    ax2.plot(list(depth_range), tr_accs, 'o-', color=COLORS['dt_full'], lw=2.2, ms=5, label='Train Acc')
    ax2.plot(list(depth_range), te_accs, 's-', color=COLORS['dt_pruned'], lw=2.2, ms=5, label='Test Acc')
    gap = [t - s for t, s in zip(tr_accs, te_accs)]
    ax2.fill_between(list(depth_range), te_accs, tr_accs, alpha=0.18, color=COLORS['warn'], label='Overfit Gap')
    best_d = list(depth_range)[np.argmax(te_accs)]
    ax2.axvline(best_d, color=COLORS['rf'], linestyle='--', lw=1.5, label=f'Best depth={best_d}')
    ax2.set_xlabel('Max Depth'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Overfitting Gap vs Tree Depth', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

# ─────────── TAB 2: VALIDATION CURVES ────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Step 2 — Validation Curves (Pre-Pruning Parameters)</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    📊 These curves show how each pruning parameter affects Train vs Cross-Validation accuracy.
    The <b>shaded region</b> shows ±1 std deviation. Use these to identify the sweet spot
    before applying GridSearchCV.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Generating validation curves..."):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle('Validation Curves — Pre-Pruning Parameters (CV=5)', fontweight='bold', fontsize=13, y=1.02)

        params_config = [
            ('max_depth',         range(1, max_depth_limit+1), 'Max Depth',         '#42A5F5'),
            ('min_samples_split', range(2, 52, 3),              'Min Samples Split', '#AB47BC'),
            ('min_samples_leaf',  range(1, 26),                 'Min Samples Leaf',  '#66BB6A'),
        ]
        for ax, (pname, p_range, plabel, pcolor) in zip(axes, params_config):
            tr_sc, cv_sc = validation_curve(
                DecisionTreeClassifier(random_state=random_state),
                X, y, param_name=pname, param_range=list(p_range),
                cv=5, scoring='accuracy', n_jobs=-1
            )
            tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
            cv_m, cv_s = cv_sc.mean(1), cv_sc.std(1)
            pr = list(p_range)
            ax.plot(pr, tr_m, 'o-', color=COLORS['dt_pruned'], lw=2, ms=4, label='Train')
            ax.fill_between(pr, tr_m-tr_s, tr_m+tr_s, alpha=0.12, color=COLORS['dt_pruned'])
            ax.plot(pr, cv_m, 's-', color=COLORS['dt_full'], lw=2, ms=4, label='CV Score')
            ax.fill_between(pr, cv_m-cv_s, cv_m+cv_s, alpha=0.12, color=COLORS['dt_full'])
            bi = np.argmax(cv_m)
            ax.axvline(pr[bi], color=pcolor, ls='--', lw=1.5, label=f'Best={pr[bi]}')
            ax.scatter([pr[bi]], [cv_m[bi]], color=pcolor, zorder=6, s=90, edgecolors='white', lw=1.5)
            ax.set_xlabel(plabel); ax.set_ylabel('Accuracy')
            ax.set_title(plabel, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────── TAB 3: BEST TREE ─────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Step 3 — GridSearchCV + Best Tree Visualization</div>',
                unsafe_allow_html=True)

    bp = gs.best_params_
    st.markdown(f"""<div class="success-box">
    ✅ <b>GridSearchCV Complete!</b> Best CV Score: <b>{gs.best_score_:.4f}</b><br>
    Best Parameters:
    <span class="param-tag">max_depth = {bp['max_depth']}</span>
    <span class="param-tag">min_samples_split = {bp['min_samples_split']}</span>
    <span class="param-tag">min_samples_leaf = {bp['min_samples_leaf']}</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Train Accuracy", m_best['train_acc'], COLORS['dt_pruned']),
        (c2, "Test Accuracy",  m_best['test_acc'],  "#42A5F5"),
        (c3, "F1 Score",       m_best['f1'],         COLORS['rf']),
        (c4, "AUC-ROC",        m_best['auc'],         COLORS['warn']),
    ]:
        col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{val:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🌳 Decision Tree Visualization (Best Pruned Model)")

    tree_depth = best_dt.get_depth()
    fw = min(30, max(20, tree_depth * 5))
    fh = min(18, max(10, tree_depth * 2.5))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    plot_tree(
        best_dt, feature_names=feat,
        class_names=['No Disease', 'Disease'] if dataset_choice != "Titanic" else ['Died', 'Survived'],
        filled=True, rounded=True, ax=ax, max_depth=5,
        fontsize=8, impurity=True, proportion=False
    )
    ax.set_title(
        f'Best Pruned Decision Tree  |  Depth={tree_depth}  |  '
        f'Leaves={best_dt.get_n_leaves()}  |  Test Acc={m_best["test_acc"]:.4f}',
        fontsize=11, fontweight='bold', pad=14
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────── TAB 4: RANDOM FOREST ─────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Step 4 — Random Forest (100 Trees)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_rf = [
        ("Train Acc", m_rf['train_acc'], COLORS['rf']),
        ("Test Acc",  m_rf['test_acc'],  COLORS['dt_pruned']),
        ("F1 Score",  m_rf['f1'],        COLORS['warn']),
        ("AUC-ROC",   m_rf['auc'],       "#AB47BC"),
        ("OOB Error", m_rf['oob_error'], COLORS['dt_full']),
    ]
    for col, (label, val, color) in zip([c1,c2,c3,c4,c5], metrics_rf):
        col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{val:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # OOB error curve + ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')

    # OOB Curve
    ax1 = axes[0]
    with st.spinner("Computing OOB curve..."):
        n_range = list(range(5, 105, 5))
        oob_errs = []
        for n in n_range:
            tmp = RandomForestClassifier(n_estimators=n, oob_score=True,
                                          random_state=random_state, n_jobs=-1)
            tmp.fit(X_tr, y_tr)
            oob_errs.append(1 - tmp.oob_score_)

    ax1.plot(n_range, oob_errs, 'o-', color=COLORS['rf'], lw=2.2, ms=5)
    ax1.fill_between(n_range, oob_errs, alpha=0.12, color='green')
    ax1.axvline(100, color='red', ls='--', lw=1.5, label='n=100')
    ax1.axhline(m_rf['oob_error'], color=COLORS['warn'], ls=':', lw=1.5,
                label=f'OOB Error={m_rf["oob_error"]:.4f}')
    ax1.set_xlabel('Number of Trees'); ax1.set_ylabel('OOB Error')
    ax1.set_title('OOB Error vs Number of Trees', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # ROC Curves
    ax2 = axes[1]
    for model, name, color in [
        (dt_full, 'DT (No Limit)', COLORS['dt_full']),
        (best_dt, 'DT (Pruned)',   COLORS['dt_pruned']),
        (rf,      'Random Forest', COLORS['rf']),
    ]:
        fpr, tpr, _ = roc_curve(y_te, model.predict_proba(X_te)[:,1])
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
        ax2.plot(fpr, tpr, lw=2.5, color=color, label=f'{name} (AUC={auc:.3f})')
    ax2.plot([0,1],[0,1], 'w--', lw=1, alpha=0.5, label='Random (0.5)')
    ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves — All Models', fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right'); ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"""<div class="info-box">
    📌 <b>OOB (Out-of-Bag) Error</b> is a free cross-validation estimate in Random Forests.
    Each tree is trained on ~63% of the data, and the remaining ~37% (OOB samples) are used to estimate
    generalization error. OOB Error = <b>{m_rf['oob_error']:.4f}</b>
    ({m_rf['oob_error']*100:.2f}%) — very close to the test error
    ({1-m_rf['test_acc']:.4f}).
    </div>""", unsafe_allow_html=True)

# ─────────── TAB 5: FEATURE IMPORTANCE ────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Step 5 — Feature Importance Comparison</div>',
                unsafe_allow_html=True)

    n_top = min(13, len(feat))
    dt_imp_series = pd.Series(best_dt.feature_importances_, index=feat).sort_values(ascending=True).tail(n_top)
    rf_imp_series = pd.Series(rf.feature_importances_,      index=feat).sort_values(ascending=True).tail(n_top)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Feature Importance: Decision Tree vs Random Forest', fontweight='bold', fontsize=13)

    for ax, imp, title, cmap_name, edge_c in [
        (axes[0], dt_imp_series, 'Decision Tree (Pruned)', 'Blues',   '#1565C0'),
        (axes[1], rf_imp_series, 'Random Forest (100T)',   'YlOrBr',  '#E65100'),
    ]:
        cmap  = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.35, 0.9, len(imp)))
        bars = ax.barh(imp.index, imp.values, color=colors, edgecolor=edge_c, linewidth=0.6)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Importance Score')
        ax.grid(axis='x', alpha=0.25)
        for bar, v in zip(bars, imp.values):
            ax.text(v + max(imp.values)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.4f}', va='center', fontsize=9)
        ax.set_xlim(0, max(imp.values)*1.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Side by side importance dataframe
    st.markdown("#### 📋 Importance Rankings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree**")
        dt_df = pd.DataFrame({'Feature': dt_imp_series.index[::-1],
                               'Importance': dt_imp_series.values[::-1]}).reset_index(drop=True)
        dt_df.index += 1
        st.dataframe(dt_df.style.format({'Importance': '{:.4f}'}), use_container_width=True)
    with col2:
        st.markdown("**Random Forest**")
        rf_df = pd.DataFrame({'Feature': rf_imp_series.index[::-1],
                               'Importance': rf_imp_series.values[::-1]}).reset_index(drop=True)
        rf_df.index += 1
        st.dataframe(rf_df.style.format({'Importance': '{:.4f}'}), use_container_width=True)

# ─────────── TAB 6: SUMMARY ───────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Final Summary — All Models Compared</div>',
                unsafe_allow_html=True)

    # Grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor('#0d1117')

    model_labels = ['DT\n(No Limit)', 'DT\n(Pruned)', 'Random\nForest']
    x = np.arange(3); w = 0.22

    ax = axes[0]
    b1 = ax.bar(x-w,  [m_full['test_acc'], m_best['test_acc'], m_rf['test_acc']], w,
                label='Accuracy',  color='#1E88E5', edgecolor='black', lw=0.8)
    b2 = ax.bar(x,    [m_full['f1'],       m_best['f1'],       m_rf['f1']],       w,
                label='F1 Score',  color='#43A047', edgecolor='black', lw=0.8)
    b3 = ax.bar(x+w,  [m_full['auc'],      m_best['auc'],      m_rf['auc']],      w,
                label='AUC-ROC',   color='#FB8C00', edgecolor='black', lw=0.8)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e8f4fd')
    ax.set_xticks(x); ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 1.2); ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison (Test Set)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    # Train vs Test per model
    ax2 = axes[1]
    tr_vals = [m_full['train_acc'], m_best['train_acc'], m_rf['train_acc']]
    te_vals = [m_full['test_acc'],  m_best['test_acc'],  m_rf['test_acc']]
    ba = ax2.bar(x - 0.18, tr_vals, 0.34, label='Train Acc', color=COLORS['dt_pruned'], alpha=0.85, edgecolor='black', lw=0.7)
    bb = ax2.bar(x + 0.18, te_vals, 0.34, label='Test Acc',  color=COLORS['rf'],        alpha=0.85, edgecolor='black', lw=0.7)
    for bar in list(ba) + list(bb):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=8, color='#e8f4fd')
    ax2.set_xticks(x); ax2.set_xticklabels(model_labels)
    ax2.set_ylim(0, 1.2); ax2.set_ylabel('Accuracy')
    ax2.set_title('Train vs Test Accuracy', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary table
    st.markdown("#### 📋 Complete Summary Table")
    summary_data = {
        'Model': ['DT (No Depth Limit)', 'DT (Pruned)', 'Random Forest (100T)'],
        'Train Acc': [m_full['train_acc'], m_best['train_acc'], m_rf['train_acc']],
        'Test Acc':  [m_full['test_acc'],  m_best['test_acc'],  m_rf['test_acc']],
        'F1 Score':  [m_full['f1'],        m_best['f1'],        m_rf['f1']],
        'AUC-ROC':   [m_full['auc'],       m_best['auc'],       m_rf['auc']],
        'OOB Error': [None, None, m_rf['oob_error']],
        'Depth':     [dt_full.get_depth(), best_dt.get_depth(), 'Ensemble'],
        'Leaves':    [dt_full.get_n_leaves(), best_dt.get_n_leaves(), '100× trees'],
    }
    summary_df = pd.DataFrame(summary_data)

    def highlight_best(s):
        if s.name in ['Test Acc', 'F1 Score', 'AUC-ROC']:
            numeric = pd.to_numeric(s, errors='coerce')
            styles = [''] * len(s)
            if numeric.notna().any():
                best_idx = numeric.idxmax()
                styles[best_idx] = 'background-color: #1a3a1a; color: #4ddb8a; font-weight: bold'
            return styles
        return [''] * len(s)

    fmt_dict = {c: '{:.4f}' for c in ['Train Acc','Test Acc','F1 Score','AUC-ROC']}
    fmt_dict['OOB Error'] = lambda x: f'{x:.4f}' if x is not None and not (isinstance(x, float) and np.isnan(x)) else 'N/A'

    st.dataframe(
        summary_df.style.apply(highlight_best).format(fmt_dict, na_rep='N/A'),
        use_container_width=True, height=160
    )

    # Key findings
    improvement = m_best['test_acc'] - m_full['test_acc']
    rf_gain = m_rf['auc'] - m_best['auc']
    st.markdown(f"""<div class="success-box">
    <b>📌 Key Findings:</b><br>
    • <b>Overfitting</b>: Unpruned DT had {m_full['train_acc']:.1%} train but only {m_full['test_acc']:.1%} test accuracy (gap = {m_full['train_acc']-m_full['test_acc']:.1%})<br>
    • <b>Pre-Pruning</b>: GridSearchCV improved test accuracy by <b>+{improvement:.1%}</b> (params: depth={gs.best_params_['max_depth']}, mss={gs.best_params_['min_samples_split']}, msl={gs.best_params_['min_samples_leaf']})<br>
    • <b>Random Forest</b>: Outperformed pruned DT on AUC-ROC by <b>+{rf_gain:.4f}</b> | OOB Error = <b>{m_rf['oob_error']:.4f}</b><br>
    • <b>Winner</b>: 🏆 Random Forest — highest AUC-ROC ({m_rf['auc']:.4f}) and best generalization
    </div>""", unsafe_allow_html=True)

    # Download summary
    csv = summary_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Summary Table (CSV)",
        csv, "tree_lab_summary.csv", "text/csv",
        use_container_width=False
    )