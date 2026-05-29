# Customer Churn Prediction | Machine Learning

## Overview
End-to-end machine learning project predicting customer churn for a telecom company. Covers full ML pipeline from EDA through model tuning.

## Dataset
- Source: IBM Telco Customer Churn (Kaggle)
- Records: 7,043 customers, 21 features
- Target: Churn (Yes/No → 1/0)
- Churn Rate: 26.5%

## Tech Stack
Python | Pandas | NumPy | Scikit-learn |
XGBoost | Matplotlib | Seaborn | Jupyter Notebook

## ML Pipeline Implemented

| Step | What Was Done |
|---|---|
| EDA | Distribution analysis, churn patterns |
| Data Cleaning | Fixed TotalCharges dtype, handled nulls |
| Encoding | Label Encoding on 16 categorical columns |
| Scaling | StandardScaler on numeric features |
| Splitting | 80/20 train-test with stratification |
| Models | Random Forest + XGBoost |
| Evaluation | Accuracy, AUC, Confusion Matrix, ROC |
| Tuning | GridSearchCV on XGBoost |

## Model Results

| Model | Accuracy | AUC |
|---|---|---|
| Random Forest | ~79% | ~0.83 |
| XGBoost (default) | ~80% | ~0.84 |
| XGBoost (tuned) | ~81% | ~0.85 |

## Key Findings
1. Month-to-month customers churn at 42% vs 3% for two-year contracts
2. High monthly charges correlate with higher churn risk
3. Electronic check payment users churn most
4. Tenure and MonthlyCharges are top predictive features

## Files
- Customer Churn.ipynb — Full notebook
- WA_Fn-UseC_-Telco-Customer-Churn.csv — Raw data
- telco_churn_cleaned.csv — Cleaned data
- confusion matric to roc curve — All major visualizations
