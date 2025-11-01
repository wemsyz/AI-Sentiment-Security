import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler (Supervised ML Task)
model = joblib.load("XGBClassifier_model.pkl")
scaler = joblib.load("scaler.pkl")


st.write("This app predicts whether a website is **Legitimate** or **Phishing** based on its attributes.")

# Collect user inputs

length_url = st.number_input("URL Length", min_value=0, max_value=500, value=50)
qty_dot_url = st.selectbox("Dots in URL", [0, 1, 2])
qty_hyphen_url = st.selectbox("Hypens in URL", [0, 1])
qty_slash_url = st.selectbox("Slashes in URL", [0, 1, 2, 3])
domain_length = st.number_input("Length of FQDN", min_value=0, max_value=255, value=50)
time_response = st.number_input("URL response time", min_value=0, max_value=120, value=12)
tls_ssl_certificate = st.selectbox("TLS/SSL certificate available", ["Yes", "No"])
qty_redirects = st.selectbox("Number of redirects", [0, 1])
url_shortened = st.selectbox("Shortened URL", ["Yes", "No"])

# Convert string inputs ("Yes"/"No") → numeric (1.0/0.0)

tls_ssl_certificate = 1.0 if tls_ssl_certificate == "Yes" else 0.0
url_shortened = 1.0 if url_shortened == "Yes" else 0.0

#  Prepare input data
input_data = {
    "length_url": [float(length_url)],
    "qty_dot_url": [float(qty_dot_url)],
    "qty_hyphen_url": [float(qty_hyphen_url)],
    "qty_slash_url": [float(qty_slash_url)],
    "domain_length": [float(domain_length)],
    "time_response": [float(time_response)],
    "tls_ssl_certificate": [tls_ssl_certificate],
    "qty_redirects": [float(qty_redirects)],
    "url_shortened": [url_shortened]
}

# Convert user inputs to DataFrame
input_df = pd.DataFrame(input_data)

#  Match input columns to training columns
expected_features = scaler.feature_names_in_  
input_df = input_df.reindex(columns=expected_features, fill_value=0)

#  Run prediction only when button is clicked
if st.button("🔍 Predict"):

    # Scale inputs and make prediction

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    # proba = model.predict_proba(scaled_input)[0][1]  # Probability of phishing


    # Display results

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"🚨 Phishing Website Detected!")
    else:
        st.success(f"✅ Legitimate Website")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")

