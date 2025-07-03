# Credit Card Approval Prediction

This project is a machine learning application that predicts whether a credit card applicant is a **good client** or **bad client**, based on their personal and financial data. It is deployed using [Streamlit](https://streamlit.io/) for interactive inference and hosted on Streamlit Cloud.

## Dataset

The dataset consists of customer demographics and credit history. It combines:

- `application_record.csv`: Contains customer demographic and income data.
- `credit_record.csv`: Contains credit payment history.

After merging and preprocessing, the dataset includes 36,457 records.

## Features

- Gender, car ownership, property ownership
- Income, number of children, family status
- Employment status, education level
- Age and years employed
- Occupation type

## Machine Learning

- **Preprocessing:**
  - One-Hot Encoding for categorical features
  - Feature scaling with `StandardScaler`
  - Handling class imbalance using SMOTE

- **Models Evaluated:**
  - Logistic Regression
  - Random Forest (best performance)
  - XGBoost
  - K-Nearest Neighbors

- **Evaluation Metrics:**
  - Precision, Recall, F1-Score
  - ROC AUC Score
  - Confusion Matrix


