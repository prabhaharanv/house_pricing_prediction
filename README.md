# 🏠 Advanced House Price Prediction (Kaggle)

This project builds an advanced regression pipeline for the Kaggle House Prices competition using extensive feature engineering and a 3-model ensemble.

📊 Cross-Validation RMSLE: **0.13455**

---

## 🧠 Problem Overview

Predict the final sale price of residential homes in Ames, Iowa using 79 explanatory variables.

Evaluation Metric: RMSLE (Root Mean Squared Log Error)

---

## 🚀 Approach

### 1️⃣ Data Preprocessing

- Cleaned inconsistent categorical labels
- Repaired corrupted values (e.g., GarageYrBlt)
- Ordinal encoding using defined category levels
- Imputation (numeric → 0, categorical → "None")

---

### 2️⃣ Feature Engineering

#### 📌 Mutual Information Feature Selection
Selected informative predictors before modeling.

#### 📌 Mathematical Transformations
- LivLotRatio
- Spaciousness
- TotalSF
- TotalBathrooms
- TotalPorchSF
- HouseAge
- YearsSinceRemodel

#### 📌 Interaction Features
- BldgType × GrLivArea

#### 📌 Neighborhood Group Statistics
- Median neighborhood living area

#### 📌 Clustering
- 20-cluster KMeans
- Cluster labels
- Distance-to-centroid features

#### 📌 PCA Components
Generated principal components from correlated spatial features.

#### 📌 Target Encoding
Implemented custom Cross-Fold M-Estimate Encoding to reduce leakage.

---

## 🤖 Models

### Model 1 — XGBoost
- 1500 estimators
- learning_rate=0.03
- max_depth=4
- regularization

### Model 2 — Gradient Boosting Regressor
- 1000 estimators
- learning_rate=0.03

### Model 3 — Ridge Regression
- StandardScaler
- alpha=10

---

## 🧮 Ensembling

Weighted blend:
- 50% XGBoost
- 30% Gradient Boosting
- 20% Ridge

Predictions trained on log(SalePrice) and exponentiated for final output.

---

## 📈 Results

5-fold Cross-Validation RMSLE:

0.13455

---

## 🛠 Tech Stack

- Python
- XGBoost
- Scikit-learn
- PCA
- KMeans
- Mutual Information
- Target Encoding
- Model Blending

---

## 💡 Key Learnings

- Feature engineering remains extremely powerful in structured ML problems.
- Clustering and group statistics add strong predictive signal.
- Cross-fold target encoding reduces leakage while preserving performance.
- Ensembling improves robustness over single-model approaches.

---
