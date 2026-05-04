# Task 1: Logistic Regression Implementation

## 📌 Task Overview
Implement Binary Logistic Regression manually using NumPy and compare with sklearn's implementation on the Breast Cancer Wisconsin dataset. Extend to multiclass classification using Iris dataset with One-vs-Rest (OvR) strategy.

## 🎯 Objectives
1. Implement sigmoid function, log-loss, and gradient descent from scratch
2. Compare manual weights vs sklearn coefficients
3. Plot decision boundary on two selected features
4. Extend to multiclass with OvR strategy
5. Display predict_proba() outputs for multiclass

## 📊 Dataset Information

### Breast Cancer Wisconsin Dataset
- **Samples:** 569
- **Features:** 30 (mean radius, texture, perimeter, area, smoothness, etc.)
- **Target:** Binary (Malignant = 0, Benign = 1)
- **Class Distribution:** 212 Malignant (37.3%), 357 Benign (62.7%)

### Iris Dataset (Multiclass)
- **Samples:** 150
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Target:** 3 classes (Setosa, Versicolor, Virginica)
- **Class Distribution:** 50 samples per class (balanced)

## 🔬 Implementation Details

### From Scratch Implementation

**Sigmoid Function:**
