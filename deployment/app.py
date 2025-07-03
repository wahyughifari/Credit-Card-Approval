import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and feature columns
model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")
st.title("Credit Risk Prediction App")
st.markdown("Predict whether a client is a **bad client** or **good client** based on their credit application data.")

# ===== USER INPUT FORM =====
st.header("Enter Client Information")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        CNT_CHILDREN = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        AGE_YEARS = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        FLAG_PHONE = st.selectbox("Has Home Phone?", ['Yes', 'No'])
    with col2:
        AMT_INCOME_TOTAL = st.number_input("Annual Income", value=1000000.0, step=100000.0)
        YEARS_EMPLOYED = st.number_input("Years Employed", value=5.0, step=0.5)
        FLAG_EMAIL = st.selectbox("Has Email?", ['Yes', 'No'])
    with col3:
        CNT_FAM_MEMBERS = st.number_input("Number of Family Members", min_value=1.0, value=2.0)
        FLAG_MOBIL = st.selectbox("Has Mobile?", ['Yes', 'No'])
        FLAG_WORK_PHONE = st.selectbox("Has Work Phone?", ['Yes', 'No'])

    # Categorical
    CODE_GENDER = st.selectbox("Gender", ['Female', 'Male'])
    FLAG_OWN_CAR = st.selectbox("Owns Car?", ['No', 'Yes'])
    FLAG_OWN_REALTY = st.selectbox("Owns Real Estate?", ['No', 'Yes'])
    NAME_INCOME_TYPE = st.selectbox("Income Type", ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'])
    NAME_EDUCATION_TYPE = st.selectbox("Education Type", ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'])
    NAME_FAMILY_STATUS = st.selectbox("Marital Status", ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'])
    NAME_HOUSING_TYPE = st.selectbox("Housing Type", ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'])
    OCCUPATION_TYPE = st.selectbox("Occupation", ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers', 'Sales staff', 'Cleaning staff',
                                                   'Cooking staff', 'Private service staff', 'Medicine staff', 'Security staff', 'High skill tech staff',
                                                   'Waiters/barmen staff', 'Low-skill Laborers', 'Secretaries', 'Realty agents', 'HR staff', 'IT staff', 'Unknown'])

    submit = st.form_submit_button("Predict")

# ===== INFERENCE =====
if submit:
    # Prepare input as a dictionary
    input_dict = {
        'CNT_CHILDREN': CNT_CHILDREN,
        'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
        'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS,
        'AGE_YEARS': AGE_YEARS,
        'YEARS_EMPLOYED': YEARS_EMPLOYED,
        'FLAG_MOBIL': 1 if FLAG_MOBIL == 'Yes' else 0,
        'FLAG_WORK_PHONE': 1 if FLAG_WORK_PHONE == 'Yes' else 0,
        'FLAG_PHONE': 1 if FLAG_PHONE == 'Yes' else 0,
        'FLAG_EMAIL': 1 if FLAG_EMAIL == 'Yes' else 0,
        'CODE_GENDER_M': 1 if CODE_GENDER == 'Male' else 0,
        'FLAG_OWN_CAR_Y': 1 if FLAG_OWN_CAR == 'Yes' else 0,
        'FLAG_OWN_REALTY_Y': 1 if FLAG_OWN_REALTY == 'Yes' else 0,
        # OHE features
        f'NAME_INCOME_TYPE_{NAME_INCOME_TYPE}': 1,
        f'NAME_EDUCATION_TYPE_{NAME_EDUCATION_TYPE}': 1,
        f'NAME_FAMILY_STATUS_{NAME_FAMILY_STATUS}': 1,
        f'NAME_HOUSING_TYPE_{NAME_HOUSING_TYPE}': 1,
        f'OCCUPATION_TYPE_{OCCUPATION_TYPE}': 1
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_dict])

    # Ensure all required features are present
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]  # Reorder columns

    # Scale numeric features
    num_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'AGE_YEARS', 'YEARS_EMPLOYED']
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    # Make prediction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.subheader("Prediction Result")
    st.success("✅ Good Client" if prediction == 0 else "❌ Bad Client")
    st.write(f"Probability of being a **Bad Client**: **{probability:.2%}**")
