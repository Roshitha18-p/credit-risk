# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1️⃣ Load saved model, scaler, and column list
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# 2️⃣ App title
st.set_page_config(page_title="Credit Risk Analysis", layout="centered")
st.title("💳 Credit Risk Analysis System")
st.write("Enter applicant details below:")

# 3️⃣ User Inputs
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])

# Optional categorical inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0","1","2","3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

# 4️⃣ Prediction button
if st.button("Predict"):

    # 4a️⃣ Create input dataframe
    input_df = pd.DataFrame({
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Gender_Male': [1 if gender=="Male" else 0],
        'Married_Yes': [1 if married=="Yes" else 0],
        'Dependents_1': [1 if dependents=="1" else 0],
        'Dependents_2': [1 if dependents=="2" else 0],
        'Dependents_3+': [1 if dependents=="3+" else 0],
        'Education_Not Graduate': [1 if education=="Not Graduate" else 0],
        'Self_Employed_Yes': [1 if self_employed=="Yes" else 0],
        'Property_Area_Semiurban': [1 if property_area=="Semiurban" else 0],
        'Property_Area_Urban': [1 if property_area=="Urban" else 0]
    })

    # 4b️⃣ Add missing columns (if any)
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # 4c️⃣ Reorder columns
    input_df = input_df[columns]

    # 4d️⃣ Scale input
    input_scaled = scaler.transform(input_df)

    # 4e️⃣ Prediction
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  # probability of approval
    risk_score = (1 - prob) * 100

    # 5️⃣ Output
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"Risk Score: {risk_score:.2f}%")

    if risk_score < 30:
        st.success("🟢 Low Risk")
    elif risk_score < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")