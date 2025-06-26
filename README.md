# Credit Card Fraud Detection 🕵️‍♀️💳

This project focuses on detecting fraudulent transactions using multiple Machine Learning models applied on imbalanced data. The techniques used to handle class imbalance include **SMOTE (Synthetic Minority Oversampling Technique)** and **Class Weights**.

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Features**: 30 total (`V1` to `V28`, `Time`, `Amount`, and `Class`)
- **Target**:  
  - `0`: Legitimate  
  - `1`: Fraudulent

## ⚙️ Project Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Class imbalance visualization  
   - Amount/Time distribution  
   - Correlation heatmap  
     ![لقطة شاشة 2025-06-26 190952](https://github.com/user-attachments/assets/70382bf2-b6ee-4d1e-ab72-01ba7b91a946)

![لقطة شاشة 2025-06-26 191038](https://github.com/user-attachments/assets/9679bda2-369f-4fc4-81e9-d4affa910057)

![لقطة شاشة 2025-06-26 191053](https://github.com/user-attachments/assets/96a9821d-cd05-40ee-9970-bad8cbe99213)



2. **Data Preprocessing**
   - Standardize `Amount` and `Time` features using `StandardScaler`
   - Drop original `Amount` and `Time`

3. **Train/Test Split**
   - 80% training / 20% testing  
   - Stratified split to maintain class ratio

4. **Handling Class Imbalance**
   - **SMOTE**: Applied to oversample minority class in training data  
   - **Class Weights**: Calculated and passed to models

5. **Models Trained**
   - Logistic Regression (LR)
   - Random Forest (RF)
   - XGBoost (XGB)
  
 ## 📊 Model Comparison

| Model | Method | Precision | Recall | F1-Score |
|-------|--------|-----------|--------|----------|
| RF    | SMOTE  | 0.8144    | 0.8061 | 0.8103   |
| RF    | Class Weights | 0.9610 | 0.7551 | 0.8457   |
| LR    | SMOTE  | 0.0581    | 0.9184 | 0.1094   |
| LR    | Class Weights | 0.0609 | 0.9184 | 0.1141   |
| XGB   | SMOTE  | 0.7311    | 0.8878 | 0.8018   |
| XGB   | Class Weights | **0.8542** | **0.8367** | **0.8454** |
![لقطة شاشة 2025-06-26 190551](https://github.com/user-attachments/assets/1e309e1c-c2ce-40b7-8133-ae9ef9c18a2d)

 
### 📁 Saved Artifacts:
- `xgboost_fraud_model.pkl` — Trained XGBoost model
- `scaler.pkl` — StandardScaler used for `Amount` and `Time`

## 🧪 Final Evaluation on New Unseen Data

| Metric | Value |
|--------|-------|
| Accuracy | 99.95% |
| Recall (Fraud) | 100% |
| Precision (Fraud) | 76.92% |
| F1-score (Fraud) | 86.96% |

![لقطة شاشة 2025-06-26 191347](https://github.com/user-attachments/assets/853e2fe6-3d74-48b9-bff6-00f157571939)


## 🚀 How to Use

```python
# Load model
import joblib
model = joblib.load("xgboost_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Preprocess new data
scaled_amount = scaler.transform(new_data['Amount'].values.reshape(-1, 1))
scaled_time = scaler.transform(new_data['Time'].values.reshape(-1, 1))
# Predict
predictions = model.predict(new_data_transformed)
