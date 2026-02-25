# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Title
st.title("💳 Credit Risk Analysis System")

# Debug: show files in directory
st.write("Files in directory:", os.listdir())

# Load model, scaler, and columns safely
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    st.success("✅ Model and files loaded successfully")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Numeric Inputs
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

# Predict button
if st.button("Predict"):

    # Create dataframe with numeric inputs
    input_data = pd.DataFrame({
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history]
    })

    # Encode categorical variables
    input_data['Gender_Male'] = 1 if gender == "Male" else 0
    input_data['Married_Yes'] = 1 if married == "Yes" else 0
    input_data['Education_Not Graduate'] = 1 if education == "Not Graduate" else 0
    input_data['Self_Employed_Yes'] = 1 if self_employed == "Yes" else 0
    input_data['Property_Area_Urban'] = 1 if property_area == "Urban" else 0
    input_data['Property_Area_Semiurban'] = 1 if property_area == "Semiurban" else 0
    input_data['Dependents_1'] = 1 if dependents == "1" else 0
    input_data['Dependents_2'] = 1 if dependents == "2" else 0
    input_data['Dependents_3+'] = 1 if dependents == "3+" else 0

    # Add remaining missing columns to match model
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match model
    input_data = input_data[columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction and probability
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    # Risk score
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