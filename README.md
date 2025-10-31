# Loan Approval Prediction  
**91% Accuracy | Random Forest + SMOTE**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YSkUd8O7rI3IHgQFmqsSp9bteztdZw4L)

---

## Dataset
- **Source**: [Loan Approval Classification Dataset][https://www.kaggle.com/datasets/laotseu/credit-risk-dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) 
- **Size**: 45,000 loan applications  
- **Target**: `loan_status` → `1` = Approved, `0` = Rejected  

---

## Overview
A **machine learning model** to predict **loan approval** based on applicant data.

- **Model**: `Random Forest`  
- **Accuracy**: **91%**  
- **F1-Score (Rejected class)**: **0.80**

---

## Key Features
| Feature | Insight |
|-------|--------|
| `previous_loan_defaults_on_file` | **Strongest predictor** – instant rejection |
| `loan_percent_income > 0.5` | High risk → engineered `high_risk` flag |
| `credit_score` & `loan_int_rate` | Key differentiators |

---

## Techniques Used
- **EDA**: Histograms, boxplots, outlier detection (IQR)
- **Feature Engineering**: `high_risk` flag
- **Preprocessing**: `LabelEncoder`, `StandardScaler`
- **Imbalance Handling**: **SMOTE + `class_weight='balanced'`**
- **Model**: `RandomForestClassifier(n_estimators=200, max_depth=15)`
- **Evaluation**: `classification_report`, confusion matrix, feature importance

---

## Results
```text
              precision    recall  f1-score   support
    0         0.96      0.92      0.94      7000
    1         0.75      0.87      0.80      2000
accuracy                           0.91      9000
