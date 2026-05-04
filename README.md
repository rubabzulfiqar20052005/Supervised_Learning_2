# Name: Rubab 
# UON (ML internship)
# Supervised Learning-II: Complete Implementation Report

## Overview
This repository contains a complete implementation of key supervised learning algorithms covered in the ML (Supervised Learning-II) module. All tasks have been successfully completed, from scratch implementations using NumPy to production-ready pipelines with scikit-learn.

## Tasks Completed

### 1. Logistic Regression from Scratch + sklearn
- **Dataset**: Breast Cancer Wisconsin (binary), Iris (multiclass)
- **Implementation**:
  - Manual Logistic Regression using NumPy (sigmoid, log-loss, gradient descent)
  - sklearn's `LogisticRegression()` for comparison
  - Coefficient comparison (manual vs sklearn)
  - Decision boundary visualization on two selected features
  - Multiclass classification using One-vs-Rest (OvR) strategy with Softmax
  - `predict_proba()` outputs for Iris dataset

### 2. Classification Metrics Dashboard
- **Dataset**: Imbalanced binary classification (Credit Card Fraud / Titanic)
- **Implementation**:
  - Manual computation of Precision, Recall, F1, Accuracy from confusion matrix
  - Verification with `classification_report()`
  - Confusion matrix heatmap visualization
  - ROC curve with AUC shading
  - Precision-Recall curve
  - Threshold analysis (0.1 to 0.9) showing Precision/Recall tradeoff
  - Summary table of all metrics at default 0.5 threshold

### 3. KNN vs Naive Bayes Comparison
- **Datasets**: Digits (or Wine Quality), 20 Newsgroups (text)
- **Implementation**:
  - KNN with K=1 to 20, elbow curve for optimal K
  - Decision boundary visualization (K=1, 5, 15) using PCA components
  - GaussianNB for feature-based classification
  - MultinomialNB with TF-IDF for text classification
  - Performance comparison table (accuracy, precision, recall, F1)

### 4. Decision Tree & Random Forest Lab
- **Dataset**: Heart Disease / Titanic
- **Implementation**:
  - Unpruned Decision Tree (overfitting demonstration)
  - Pre-pruning with `max_depth`, `min_samples_split`, `min_samples_leaf`
  - GridSearchCV for hyperparameter tuning
  - Validation curves and `plot_tree()` visualization
  - RandomForestClassifier with 100 trees
  - Feature importance bar charts (both models)
  - OOB error reporting
  - Final comparison table

### 5. SVM Kernel Comparison Lab
- **Datasets**: `make_moons`, `make_circles` (synthetic), Breast Cancer
- **Implementation**:
  - Decision boundary visualization for Linear, Polynomial (deg 2 & 3), RBF kernels
  - StandardScaler application for Breast Cancer dataset
  - GridSearchCV for C and gamma tuning (RBF kernel)
  - Heatmap of GridSearch scores (C vs gamma)
  - Model comparison: SVM vs Logistic Regression vs KNN vs Random Forest
  - Final comparison table across all five classifiers

### 6. End-to-End Classification Pipeline (Final Project)
- **Dataset**: Kaggle dataset (Titanic Survival / Customer Churn / Heart Disease)
- **Complete Pipeline**:
  - Load and inspect data
  - Exploratory Data Analysis (EDA) with key plots
  - Data cleaning (missing values, outliers, inconsistencies)
  - Preprocessing (scaling, encoding, feature engineering)
  - Train/test split
  - Train 6 models: Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, SVM
  - Evaluation metrics: Accuracy, Precision, Recall, F1, AUC-ROC
  - Model comparison summary table
  - Save best model using `pickle`
  - Predict on 5 new unseen samples
  - README.txt with dataset summary, EDA findings, preprocessing decisions, and best model analysis

## Technologies Used
- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (all major modules)
- Pickle (model serialization)
- Imbalanced-learn (for imbalanced datasets)

## Key Insights Learned
- Logistic Regression requires feature scaling and handles probability boundaries well
- Accuracy is misleading for imbalanced datasets; Precision-Recall curves are better
- KNN is sensitive to feature scaling and suffers from curse of dimensionality
- Naive Bayes works surprisingly well for text despite independence assumption
- Decision Trees overfit easily; pruning and Random Forests reduce variance
- SVM with RBF kernel is powerful but requires careful C/gamma tuning
- End-to-end pipelines ensure reproducibility and production readiness

## How to Run
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run each task notebook in order
4. Check `models/` folder for saved `.pkl` best models
5. Review `README.txt` inside final project folder for dataset-specific details

## Results Summary
- **Best performer overall**: Random Forest / SVM (depending on dataset)
- **Fastest training**: Naive Bayes / Logistic Regression
- **Best for imbalanced data**: Random Forest with class_weight tuning
- **Best for text**: MultinomialNB with TF-IDF
- **Interpretable models**: Logistic Regression, Decision Tree

## Author
Completed as part of ML (Supervised Learning-II) module  
Date: 30 April 2026 – 04 May 2026
