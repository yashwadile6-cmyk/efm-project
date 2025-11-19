# app.py
# 🖥️ A simple web dashboard for your project

import streamlit as st
import pandas as pd
import joblib
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="India Unemployment Prediction",
    page_icon="📊",
    layout="centered"
)

# --- Load Model ---
# Load the best model you saved (Random Forest)
try:
    model = joblib.load("unemployment_rf_model.pkl")
except FileNotFoundError:
    st.error("Model file (unemployment_rf_model.pkl) not found. Please run the main script first to generate it.")
    st.stop()


# --- App UI ---
st.title("📊 Unemployment Rate Prediction for India")
st.write("This app uses a *Random Forest Regressor* to predict the unemployment rate based on the year and month.")
st.write("---")

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Enter Year", 
        min_value=2020, 
        max_value=2030, 
        value=2025,
        help="Select the year for prediction."
    )

with col2:
    month = st.selectbox(
        "Enter Month",
        options=range(1, 13),
        format_func=lambda x: pd.to_datetime(f'2024-{x}-01').strftime('%B'), # Show month names
        help="Select the month for prediction."
    )

# --- Prediction Button ---
if st.button("🚀 Predict Rate", type="primary"):
    
    # Create input features
    input_features = [[year, month]]
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Display result with a spinner
    with st.spinner('Calculating...'):
        time.sleep(1) # Just to make it feel impressive
    
    st.success(f"**Predicted Unemployment Rate:**")
    
    # Using columns for a cleaner layout
    res_col1, res_col2 = st.columns([1,2])
    with res_col1:
        st.metric(
            label=f"for {pd.to_datetime(f'{year}-{month}-01').strftime('%B, %Y')}",
            value=f"{prediction:.2f}%"
        )
    with res_col2:
        st.write(f"The model predicts an estimated unemployment rate of **{prediction:.2f}%** for the selected period.")

st.write("---")
st.caption("This tool is based on a machine learning model trained on historical data from 'Unemployment in India.csv'.")