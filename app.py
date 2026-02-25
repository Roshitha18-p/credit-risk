import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("💳 Credit Risk Analysis System")

st.write("Enter applicant details below:")

# Inputs
income = st.number_input("Applicant Income", min_value=0)
co_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])

# Button
if st.button("Predict"):

    # Create empty dataframe with all columns
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Fill required columns (only if they exist)
    if 'ApplicantIncome' in input_df.columns:
        input_df['ApplicantIncome'] = income

    if 'CoapplicantIncome' in input_df.columns:
        input_df['CoapplicantIncome'] = co_income

    if 'LoanAmount' in input_df.columns:
        input_df['LoanAmount'] = loan_amount

    if 'Loan_Amount_Term' in input_df.columns:
        input_df['Loan_Amount_Term'] = loan_term

    if 'Credit_History' in input_df.columns:
        input_df['Credit_History'] = credit_history

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    risk_score = probability[0][1] * 100

    # Output
    st.subheader("Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"### Risk Score: {risk_score:.2f}%")

    # Risk category
    if risk_score < 40:
        st.success("Low Risk")
    elif risk_score < 70:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")