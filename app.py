import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import uuid

# Load model and label encoders
model = joblib.load('hybrid_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit app
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it's fraudulent.")

# Input fields
trans_date_trans_time = st.text_input("Transaction Date and Time (YYYY-MM-DD HH:MM:SS)", "2023-01-01 12:00:00")
amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.00, step=0.01)
category = st.selectbox("Transaction Category", label_encoders['category'].classes_)
merchant = st.selectbox("Merchant", label_encoders['merchant'].classes_)
city = st.selectbox("City", label_encoders['city'].classes_)
state = st.selectbox("State", label_encoders['state'].classes_)
zip_code = st.number_input("Zip Code", min_value=10000, max_value=99999, value=10001)
lat = st.number_input("Cardholder Latitude", min_value=-90.0, max_value=90.0, value=40.7128)
long = st.number_input("Cardholder Longitude", min_value=-180.0, max_value=180.0, value=-74.0060)
city_pop = st.number_input("City Population", min_value=1, value=100000)
merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=40.7128)
merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-74.0060)
gender = st.selectbox("Gender", label_encoders['gender'].classes_)

# Prediction button
if st.button("Predict"):
    try:
        # Process input
        trans_dt = pd.to_datetime(trans_date_trans_time)
        trans_hour = trans_dt.hour
        trans_day = trans_dt.dayofweek

        # Encode categorical inputs
        input_data = {
            'merchant': label_encoders['merchant'].transform([merchant])[0],
            'category': label_encoders['category'].transform([category])[0],
            'amt': amt,
            'gender': label_encoders['gender'].transform([gender])[0],
            'city': label_encoders['city'].transform([city])[0],
            'state': label_encoders['state'].transform([state])[0],
            'zip': zip_code,
            'lat': lat,
            'long': long,
            'city_pop': city_pop,
            'merch_lat': merch_lat,
            'merch_long': merch_long,
            'trans_hour': trans_hour,
            'trans_day': trans_day
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure column order matches training data
        feature_order = ['merchant', 'category', 'amt', 'gender', 'city', 'state', 'zip', 'lat', 'long',
                         'city_pop', 'merch_lat', 'merch_long', 'trans_hour', 'trans_day']
        input_df = input_df[feature_order]

        # Make prediction
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        # Display results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"Fraudulent Transaction! (Probability: {prob:.2%})")
        else:
            st.success(f"Legitimate Transaction (Probability of Fraud: {prob:.2%})")

    except Exception as e:
        st.error(f"Error processing input: {str(e)}")

# Instructions
st.markdown("""
### Instructions
1. Enter the transaction details in the fields above.
2. Click the "Predict" button to get the fraud prediction.
3. The model will indicate if the transaction is fraudulent or legitimate, along with the probability of fraud.
""")