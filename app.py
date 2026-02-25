# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("💳 Credit Risk Analysis System")

st.write("Enter applicant details below:")

# Inputs
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])

# When button clicked
if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history]
    })

    # Add missing columns
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns
    input_data = input_data[columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    # FIXED risk score
    risk_score = (1 - prob[0][1]) * 100

    # Output
    st.subheader("Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"Risk Score: {risk_score:.2f}%")

    # Risk level
    if risk_score < 30:
        st.success("🟢 Low Risk")
    elif risk_score < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")