"""
╔══════════════════════════════════════════════════════════════════════╗
║        TREE MODELS LAB — Heart Disease / Titanic Dataset            ║
║  Decision Tree  →  Pre-Pruning  →  Random Forest  →  Comparison     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

os.makedirs('outputs', exist_ok=True)

RANDOM_STATE = 42
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'dt_full': '#E53935', 'dt_pruned': '#1E88E5', 'rf': '#43A047'}

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════════
def load_heart_disease():
    """Try multiple sources for Heart Disease dataset."""
    # --- Source 1: Direct URL (works when internet available) ---
    urls = [
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv",
        "https://storage.googleapis.com/kagglestorage/heart.csv",
    ]
    for url in urls:
        try:
            import urllib.request
            urllib.request.urlretrieve(url, '/tmp/_heart.csv')
            df = pd.read_csv('/tmp/_heart.csv')
            if 'target' in df.columns and len(df) > 100:
                df.columns = [c.strip().lower() for c in df.columns]
                print(f"  ✓ Heart Disease loaded from URL ({df.shape[0]} rows)")
                return df, 'target', 'Heart Disease (UCI Cleveland)'
        except Exception:
            pass

    # --- Source 2: OpenML ---
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml('heart-statlog', version=1, as_frame=True, parser='auto')
        df = data.frame.copy()
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['class'].astype(str))
        df.drop('class', axis=1, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        print(f"  ✓ Heart Disease loaded from OpenML ({df.shape[0]} rows)")
        return df, 'target', 'Heart Statlog (OpenML)'
    except Exception:
        pass

    # --- Fallback: Synthetic Heart Disease (realistic distributions) ---
    print("  ⚠  Internet unavailable — generating realistic Heart Disease dataset...")
    np.random.seed(RANDOM_STATE)
    n = 303
    age = np.random.normal(54, 9, n).clip(29, 77).astype(int)
    sex = np.random.binomial(1, 0.68, n)
    cp = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.random.normal(131, 17, n).clip(94, 200).astype(int)
    chol = np.random.normal(246, 52, n).clip(126, 564).astype(int)
    fbs = np.random.binomial(1, 0.15, n)
    restecg = np.random.choice([0, 1, 2], n, p=[0.50, 0.01, 0.49])
    thalach = np.random.normal(150, 23, n).clip(71, 202).astype(int)
    exang = np.random.binomial(1, 0.33, n)
    oldpeak = np.random.exponential(1.0, n).clip(0, 6.2).round(1)
    slope = np.random.choice([0, 1, 2], n, p=[0.08, 0.46, 0.46])
    ca = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([1, 2, 3], n, p=[0.05, 0.55, 0.40])

    # Realistic target based on features
    risk = (
        0.04 * (age - 50) +
        0.3 * sex +
        0.4 * (cp == 0).astype(int) -
        0.02 * (thalach - 150) +
        0.5 * exang +
        0.3 * oldpeak +
        0.3 * ca +
        0.2 * (thal == 3).astype(int) +
        np.random.normal(0, 0.3, n)
    )
    target = (risk > np.median(risk)).astype(int)

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
        'thal': thal, 'target': target
    })
    print(f"  ✓ Synthetic Heart Disease generated ({df.shape[0]} rows)")
    return df, 'target', 'Heart Disease (Synthetic — realistic distributions)'


print("=" * 65)
print("  TREE MODELS LAB  |  Heart Disease Dataset")
print("=" * 65)

print("\n[STEP 1] Loading Dataset...")
df, target_col, dataset_name = load_heart_disease()

# Encode any remaining object columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df = df.apply(pd.to_numeric, errors='coerce').dropna()

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Dataset : {dataset_name}")
print(f"  Shape   : {df.shape}  |  Features: {len(feature_names)}")
print(f"  Split   : Train={len(X_train)}  Test={len(X_test)}")
print(f"  Classes : {y.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════
# 2. DECISION TREE — NO DEPTH LIMIT (Observe Overfitting)
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Decision Tree — No Depth Limit (Overfitting)...")

dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_full.fit(X_train, y_train)

dt_full_train_acc = accuracy_score(y_train, dt_full.predict(X_train))
dt_full_test_acc  = accuracy_score(y_test,  dt_full.predict(X_test))
dt_full_f1        = f1_score(y_test, dt_full.predict(X_test), zero_division=0)
dt_full_auc       = roc_auc_score(y_test, dt_full.predict_proba(X_test)[:, 1])

print(f"  Depth   : {dt_full.get_depth()}")
print(f"  Leaves  : {dt_full.get_n_leaves()}")
print(f"  Train Acc: {dt_full_train_acc:.4f}  |  Test Acc: {dt_full_test_acc:.4f}")
print(f"  → GAP (Train-Test): {dt_full_train_acc - dt_full_test_acc:.4f}  ← OVERFITTING!")

# Plot: Train vs Test accuracy bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Step 2: Observing Overfitting in Unpruned Decision Tree', fontsize=14, fontweight='bold')

ax = axes[0]
bars = ax.bar(['Train Accuracy', 'Test Accuracy'],
               [dt_full_train_acc, dt_full_test_acc],
               color=[COLORS['dt_full'], '#90CAF9'], edgecolor='black', width=0.45)
ax.set_ylim(0, 1.15)
ax.set_title('Unpruned DT: Train vs Test Accuracy', fontweight='bold')
ax.set_ylabel('Accuracy')
for bar, val in zip(bars, [dt_full_train_acc, dt_full_test_acc]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', fontweight='bold', fontsize=12)
ax.axhline(dt_full_test_acc, color='red', linestyle='--', alpha=0.5, label='Test Level')
ax.legend()
ax.grid(axis='y', alpha=0.4)

# Plot: depth complexity curve
ax2 = axes[1]
depth_range = range(1, min(25, dt_full.get_depth() + 2))
train_accs, test_accs = [], []
for d in depth_range:
    tmp = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    tmp.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, tmp.predict(X_train)))
    test_accs.append(accuracy_score(y_test,  tmp.predict(X_test)))
ax2.plot(depth_range, train_accs, 'b-o', markersize=5, label='Train Acc', linewidth=2)
ax2.plot(depth_range, test_accs,  'r-o', markersize=5, label='Test Acc',  linewidth=2)
ax2.fill_between(depth_range,
                  [t - s for t, s in zip(train_accs, test_accs)],
                  [0]*len(depth_range), alpha=0.08, color='orange', label='Overfit Gap')
ax2.set_xlabel('Max Depth', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Overfitting vs Depth', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/01_overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/01_overfitting_analysis.png")

# ══════════════════════════════════════════════════════════════════════
# 3. VALIDATION CURVES (Pre-Pruning Parameters)
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Validation Curves for Pre-Pruning...")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle('Validation Curves — Decision Tree Pre-Pruning Parameters\n(CV=5, Blue=Train, Red=CV Score)',
             fontsize=14, fontweight='bold')

pruning_params = [
    ('max_depth',         range(1, 21),    'Max Depth',          '#1565C0'),
    ('min_samples_split', range(2, 52, 2), 'Min Samples Split',  '#6A1B9A'),
    ('min_samples_leaf',  range(1, 26),    'Min Samples Leaf',   '#2E7D32'),
]

for ax, (param, p_range, label, color) in zip(axes, pruning_params):
    train_sc, val_sc = validation_curve(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        X, y,
        param_name=param, param_range=list(p_range),
        cv=5, scoring='accuracy', n_jobs=-1
    )
    tr_mean, tr_std = train_sc.mean(1), train_sc.std(1)
    cv_mean, cv_std = val_sc.mean(1),   val_sc.std(1)

    ax.plot(list(p_range), tr_mean, 'b-o', markersize=4, label='Train', linewidth=2.2)
    ax.fill_between(list(p_range), tr_mean - tr_std, tr_mean + tr_std, alpha=0.12, color='blue')
    ax.plot(list(p_range), cv_mean, 'r-s', markersize=4, label='CV Score', linewidth=2.2)
    ax.fill_between(list(p_range), cv_mean - cv_std, cv_mean + cv_std, alpha=0.12, color='red')

    best_idx = np.argmax(cv_mean)
    ax.axvline(list(p_range)[best_idx], color=color, linestyle='--', alpha=0.8,
               label=f'Best={list(p_range)[best_idx]}')
    ax.scatter([list(p_range)[best_idx]], [cv_mean[best_idx]],
               color=color, zorder=5, s=80)

    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Validation Curve:\n{label}', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(max(0, min(tr_mean.min(), cv_mean.min()) - 0.05), 1.05)

plt.tight_layout()
plt.savefig('outputs/02_validation_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/02_validation_curves.png")

# ══════════════════════════════════════════════════════════════════════
# 4. GRIDSEARCHCV — BEST DECISION TREE (Pre-Pruning)
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 4] GridSearchCV — Pre-Pruning Best Parameters...")

param_grid = {
    'max_depth':          [3, 4, 5, 6, 7, 8, 10, None],
    'min_samples_split':  [2, 5, 10, 20, 30],
    'min_samples_leaf':   [1, 2, 4, 6, 8],
}

grid_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, refit=True
)
grid_cv.fit(X_train, y_train)

best_dt      = grid_cv.best_estimator_
best_params  = grid_cv.best_params_
best_cv_acc  = grid_cv.best_score_

dt_best_train_acc = accuracy_score(y_train, best_dt.predict(X_train))
dt_best_test_acc  = accuracy_score(y_test,  best_dt.predict(X_test))
dt_best_f1        = f1_score(y_test, best_dt.predict(X_test), zero_division=0)
dt_best_auc       = roc_auc_score(y_test, best_dt.predict_proba(X_test)[:, 1])

print(f"  Best Params  : {best_params}")
print(f"  Best CV Acc  : {best_cv_acc:.4f}")
print(f"  Pruned DT    : Train={dt_best_train_acc:.4f}  |  Test={dt_best_test_acc:.4f}")
print(f"  Depth reduced: {dt_full.get_depth()} → {best_dt.get_depth()}")

# ══════════════════════════════════════════════════════════════════════
# 5. VISUALISE BEST TREE
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Visualising Best Decision Tree...")

tree_depth = best_dt.get_depth()
fig_h = max(10, tree_depth * 2.5)
fig_w = max(22, tree_depth * 5)
fig_w = min(fig_w, 36)
fig_h = min(fig_h, 22)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
plot_tree(
    best_dt,
    feature_names=feature_names,
    class_names=['No Disease', 'Disease'],
    filled=True, rounded=True,
    ax=ax, max_depth=5,
    fontsize=9, impurity=True,
    proportion=False
)
ax.set_title(
    f'Best Decision Tree (Depth={tree_depth}) — Pruned via GridSearchCV\n'
    f'Params: max_depth={best_params["max_depth"]}  '
    f'min_samples_split={best_params["min_samples_split"]}  '
    f'min_samples_leaf={best_params["min_samples_leaf"]}  |  '
    f'Test Acc={dt_best_test_acc:.4f}',
    fontsize=12, fontweight='bold', pad=20
)
plt.tight_layout()
plt.savefig('outputs/03_best_tree_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/03_best_tree_plot.png")

# ══════════════════════════════════════════════════════════════════════
# 6. RANDOM FOREST — 100 TREES
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Training Random Forest (100 Trees, OOB=True)...")

rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc  = accuracy_score(y_test,  rf.predict(X_test))
rf_f1        = f1_score(y_test, rf.predict(X_test), zero_division=0)
rf_auc       = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
rf_oob_score = rf.oob_score_
rf_oob_error = 1 - rf_oob_score

print(f"  Train Acc : {rf_train_acc:.4f}")
print(f"  Test Acc  : {rf_test_acc:.4f}")
print(f"  OOB Score : {rf_oob_score:.4f}  |  OOB Error: {rf_oob_error:.4f}")
print(f"  F1 Score  : {rf_f1:.4f}")
print(f"  AUC-ROC   : {rf_auc:.4f}")

# OOB error vs n_estimators plot
print("  → Plotting OOB Error curve...")
oob_errors = []
for n in range(10, 105, 5):
    tmp_rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                     random_state=RANDOM_STATE, n_jobs=-1)
    tmp_rf.fit(X_train, y_train)
    oob_errors.append(1 - tmp_rf.oob_score_)

fig, ax = plt.subplots(figsize=(9, 5))
n_range = list(range(10, 105, 5))
ax.plot(n_range, oob_errors, 'g-o', markersize=5, linewidth=2.2, color=COLORS['rf'])
ax.fill_between(n_range, oob_errors, alpha=0.12, color='green')
ax.axvline(100, color='red', linestyle='--', linewidth=1.5, label='n=100 (final)')
ax.axhline(rf_oob_error, color='orange', linestyle=':', linewidth=1.5,
           label=f'OOB Error at 100 = {rf_oob_error:.4f}')
ax.set_xlabel('Number of Trees', fontsize=12)
ax.set_ylabel('OOB Error', fontsize=12)
ax.set_title('Random Forest: OOB Error vs Number of Trees', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_oob_error_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/04_oob_error_curve.png")

# ══════════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Feature Importance Bar Charts...")

n_top = min(13, len(feature_names))
dt_imp = pd.Series(best_dt.feature_importances_, index=feature_names).sort_values(ascending=True).tail(n_top)
rf_imp = pd.Series(rf.feature_importances_,      index=feature_names).sort_values(ascending=True).tail(n_top)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Feature Importance: Decision Tree vs Random Forest', fontsize=14, fontweight='bold')

# Decision Tree
cmap_dt = plt.cm.Blues(np.linspace(0.35, 0.85, len(dt_imp)))
bars = axes[0].barh(dt_imp.index, dt_imp.values, color=cmap_dt, edgecolor='#1565C0', linewidth=0.6)
axes[0].set_title(f'Decision Tree (Pruned)\nTop {n_top} Features', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Importance (Gini Impurity Decrease)', fontsize=11)
axes[0].grid(axis='x', alpha=0.3)
for bar, val in zip(bars, dt_imp.values):
    axes[0].text(val + max(dt_imp.values)*0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
axes[0].set_xlim(0, max(dt_imp.values) * 1.18)

# Random Forest
cmap_rf = plt.cm.Oranges(np.linspace(0.35, 0.85, len(rf_imp)))
bars2 = axes[1].barh(rf_imp.index, rf_imp.values, color=cmap_rf, edgecolor='#E65100', linewidth=0.6)
axes[1].set_title(f'Random Forest (100 Trees)\nTop {n_top} Features', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Importance (Mean Decrease Impurity)', fontsize=11)
axes[1].grid(axis='x', alpha=0.3)
for bar, val in zip(bars2, rf_imp.values):
    axes[1].text(val + max(rf_imp.values)*0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
axes[1].set_xlim(0, max(rf_imp.values) * 1.18)

plt.tight_layout()
plt.savefig('outputs/05_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/05_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════
# 8. METRICS COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 8] Metrics Comparison Chart...")

models_names = ['DT\n(No Limit)', 'DT\n(Pruned)', 'Random\nForest']
acc_vals  = [dt_full_test_acc, dt_best_test_acc, rf_test_acc]
f1_vals   = [dt_full_f1,       dt_best_f1,       rf_f1]
auc_vals  = [dt_full_auc,      dt_best_auc,      rf_auc]

x = np.arange(len(models_names))
w = 0.22

fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w, acc_vals, w, label='Accuracy',  color='#1E88E5', edgecolor='black', linewidth=0.8)
b2 = ax.bar(x,     f1_vals,  w, label='F1 Score',  color='#43A047', edgecolor='black', linewidth=0.8)
b3 = ax.bar(x + w, auc_vals, w, label='AUC-ROC',   color='#FB8C00', edgecolor='black', linewidth=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: Accuracy | F1 Score | AUC-ROC\n(Test Set Performance)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_names, fontsize=12)
ax.set_ylim(0, 1.18)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Annotate winner
best_model_idx = np.argmax(auc_vals)
ax.annotate('🏆 Best AUC', xy=(x[best_model_idx] + w, auc_vals[best_model_idx]),
            xytext=(x[best_model_idx] + w + 0.3, auc_vals[best_model_idx] + 0.07),
            fontsize=10, color='darkorange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkorange'))

plt.tight_layout()
plt.savefig('outputs/06_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/06_metrics_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# 9. CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices (Test Set)', fontsize=14, fontweight='bold')

for ax, model, name, color in [
    (axes[0], dt_full,  'DT (No Limit)', COLORS['dt_full']),
    (axes[1], best_dt,  'DT (Pruned)',   COLORS['dt_pruned']),
    (axes[2], rf,       'Random Forest', COLORS['rf']),
]:
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                linewidths=1, linecolor='white')
    ax.set_title(name, fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/07_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/07_confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════════
# 10. FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL SUMMARY TABLE")
print("=" * 65)

summary = pd.DataFrame({
    'Model': [
        'Decision Tree (No Limit)',
        'Decision Tree (Pruned)',
        'Random Forest (100 Trees)'
    ],
    'Train Acc': [f'{dt_full_train_acc:.4f}', f'{dt_best_train_acc:.4f}', f'{rf_train_acc:.4f}'],
    'Test Acc':  [f'{dt_full_test_acc:.4f}',  f'{dt_best_test_acc:.4f}',  f'{rf_test_acc:.4f}'],
    'F1 Score':  [f'{dt_full_f1:.4f}',        f'{dt_best_f1:.4f}',        f'{rf_f1:.4f}'],
    'AUC-ROC':   [f'{dt_full_auc:.4f}',       f'{dt_best_auc:.4f}',       f'{rf_auc:.4f}'],
    'OOB Error': ['N/A',                       'N/A',                       f'{rf_oob_error:.4f}'],
    'Tree Depth': [str(dt_full.get_depth()),   str(best_dt.get_depth()),    'N/A (Ensemble)'],
    'Leaves':    [str(dt_full.get_n_leaves()), str(best_dt.get_n_leaves()), '100 × trees'],
})
print(summary.to_string(index=False))
summary.to_csv('outputs/summary_table.csv', index=False)
print("\n  → Saved: outputs/summary_table.csv")

print("\n[ANALYSIS]")
print(f"  Overfitting Gap (DT Full): {dt_full_train_acc - dt_full_test_acc:.4f}")
print(f"  Improvement (Pruning):     +{dt_best_test_acc - dt_full_test_acc:+.4f}")
print(f"  RF vs Pruned DT (AUC):     +{rf_auc - dt_best_auc:+.4f}")
print(f"  OOB Error (RF):            {rf_oob_error:.4f}  ({rf_oob_error*100:.2f}%)")

print("\n" + "=" * 65)
print("  ✅  ALL OUTPUTS SAVED IN outputs/ FOLDER")
print("  📊  Run Streamlit: streamlit run app.py")
print("=" * 65)