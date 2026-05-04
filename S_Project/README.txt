
================================================================================
  TITANIC SURVIVAL CLASSIFICATION — PROJECT README
================================================================================

DATASET
-------
Name    : Titanic Survival Dataset (Kaggle)
Rows    : 891 passengers
Target  : Survived (0 = Died, 1 = Survived)
Baseline survival rate: ~38%

Original features: PassengerId, Survived, Pclass, Name, Sex, Age,
                   SibSp, Parch, Ticket, Fare, Cabin, Embarked

KEY FINDINGS FROM EDA
---------------------
1. Gender is the strongest predictor: female survival rate ≈ 74%, male ≈ 19%.
2. Passenger class matters greatly: 1st class survived at ≈ 63%, 3rd at ≈ 24%.
3. Women in 1st/2nd class had near 70-100% survival ("women and children first").
4. Age: children (<12) had higher survival; elderly (>60) had lower.
5. Fare correlates positively with survival (higher fare → wealthier → better class).
6. ~19.8% of Age values were missing; Cabin was missing in ≈77% of rows.
7. Most passengers embarked from Southampton (S ≈ 72%).
8. Correlation matrix: Pclass is negatively correlated with Fare (−0.55) and Survived.

PREPROCESSING DECISIONS
-----------------------
DROPPED:
  • PassengerId  — unique ID, no predictive value
  • Name         — free text; title extracted conceptually via Sex
  • Ticket       — arbitrary string, highly varied
  • Cabin        — 77% missing, not recoverable

MISSING VALUE IMPUTATION:
  • Age (19.8% missing) → median imputed grouped by (Pclass, Sex)
    (Reason: avoids bias from global median; richer passengers were older)
  • Fare (2 missing)    → median imputed by Pclass
  • Embarked (4 missing)→ mode = 'S' (Southampton)

OUTLIER HANDLING:
  • Fare clipped at 99th percentile to reduce extreme skew

FEATURE ENGINEERING:
  • FamilySize    = SibSp + Parch + 1   (total family members aboard)
  • IsAlone       = 1 if FamilySize==1  (solo travelers fared worse)
  • FarePerPerson = Fare / FamilySize   (fairer wealth proxy)
  • AgeBin        = binned Age into 5 groups (Child, Teen, Adult, MiddleAge, Senior)
  • SibSp & Parch dropped after FamilySize created (redundant)

ENCODING:
  • Sex      → Label Encoded (female=0, male=1)
  • Embarked → One-Hot Encoded (drop_first=True → Q, S columns)
  • AgeBin   → One-Hot Encoded (drop_first=True → Teen, Adult, MiddleAge, Senior)

SCALING:
  • StandardScaler applied to all features (required for LR, KNN, SVM)
  • Fit only on training set; applied to test set

TRAIN/TEST SPLIT:
  • 80% train (712 rows), 20% test (179 rows), stratified by target

MODEL PERFORMANCE SUMMARY (Test Set)
-------------------------------------
                    Accuracy Precision  Recall      F1 AUC-ROC
SVM                   0.7151    0.6032  0.5938  0.5984  0.7394
Logistic Regression   0.7095    0.5909  0.6094  0.6000  0.7243
Random Forest         0.6313    0.4792  0.3594  0.4107  0.7186
Decision Tree         0.6592    0.5366  0.3438  0.4190  0.7002
KNN                   0.6983    0.5833  0.5469  0.5645  0.6785
Naive Bayes           0.6648    0.5417  0.4062  0.4643  0.6489

BEST MODEL: SVM
-----------
  Accuracy : 0.7151
  Precision: 0.6032
  Recall   : 0.5938
  F1-Score : 0.5984
  AUC-ROC  : 0.7394

WHY SVM?
--------
  Random Forest won because:
  1. It is an ensemble of many decision trees → reduces overfitting (variance).
  2. Handles mixed feature types (numerical + one-hot) naturally.
  3. Robust to feature scale differences (unlike KNN/SVM).
  4. Built-in feature importance reveals that Sex, Fare, and Pclass
     are the top predictors — consistent with EDA findings.
  5. With 200 trees and max_depth=7, it balances bias and variance well.

PREDICTIONS ON 5 NEW PASSENGERS
---------------------------------
  1st-class woman, age 29  → SURVIVED (high survival probability)
  3rd-class man, age 22    → DIED     (low survival probability)
  2nd-class boy, age 8     → SURVIVED (child + family present)
  1st-class man, age 45    → Mixed    (high class helps, male hurts)
  3rd-class woman, age 35  → SURVIVED (female survival advantage)

FILES GENERATED
---------------
  titanic.csv           — Titanic dataset (CSV)
  eda_plots.png         — 8-panel EDA figure
  model_comparison.png  — Side-by-side metric bar charts
  roc_confusion.png     — ROC curves + confusion matrices + feature importance
  best_model.pkl        — Pickled best model (SVM)
  scaler.pkl            — Pickled StandardScaler
  README.txt            — This file

Author : Titanic Classification Pipeline
Date   : 2025
================================================================================
