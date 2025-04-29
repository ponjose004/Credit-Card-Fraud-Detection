
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import streamlit as st
import pickle
import os

# Function to preprocess datetime column
def preprocess_datetime(df, column='trans_date_trans_time'):
    df[column] = pd.to_datetime(df[column])
    df['hour'] = df[column].dt.hour
    df['day_of_week'] = df[column].dt.dayofweek
    df['month'] = df[column].dt.month
    return df.drop(columns=[column])

# Function to load and preprocess data
def load_and_preprocess_data(train_path, test_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Combine for consistent preprocessing
    df = pd.concat([train_df, test_df], axis=0)
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Preprocess datetime column
    if 'trans_date_trans_time' in df.columns:
        df = preprocess_datetime(df, 'trans_date_trans_time')
    
    # Define categorical and numerical columns (adjust based on your dataset)
    categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']  # Example
    numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'hour', 'day_of_week', 'month']
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    existing_numerical_cols = [col for col in numerical_cols if col in df.columns]
    if existing_numerical_cols:
        df[existing_numerical_cols] = scaler.fit_transform(df[existing_numerical_cols])
    
    # Split back into train and test
    train_df = df.iloc[:len(train_df)]
    test_df = df.iloc[len(train_df):]
    
    return train_df, test_df, label_encoders, scaler

# Function to train models
def train_models(X_train, y_train):
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    # Train individual models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    # Create hybrid model (Voting Classifier)
    hybrid_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'
    )
    hybrid_model.fit(X_train, y_train)
    
    return rf_model, xgb_model, hybrid_model

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

# Streamlit app
def main():
    st.title("Credit Card Fraud Detection")
    st.write("Enter transaction details to predict if it's fraudulent.")
    
    # Load and preprocess data
    train_path = "FraudTrain.csv"
    test_path = "FraudTest.csv"
    
    if not os.path.exists('hybrid_model.pkl'):
        train_df, test_df, label_encoders, scaler = load_and_preprocess_data(train_path, test_path)
        
        # Prepare features and target
        feature_cols = [col for col in train_df.columns if col not in ['is_fraud', 'trans_date_trans_time', 'first', 'last', 'street', 'dob', 'trans_num']]  # Exclude irrelevant columns
        X_train = train_df[feature_cols]
        y_train = train_df['is_fraud']
        X_test = test_df[feature_cols]
        y_test = test_df['is_fraud']
        
        # Train models
        rf_model, xgb_model, hybrid_model = train_models(X_train, y_train)
        
        # Save models and preprocessing objects
        with open('hybrid_model.pkl', 'wb') as f:
            pickle.dump(hybrid_model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # Evaluate models
        models = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'Hybrid Model': hybrid_model}
        evaluate_models(models, X_test, y_test)
    else:
        # Load models and preprocessing objects
        with open('hybrid_model.pkl', 'rb') as f:
            hybrid_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    
    # User input form
    st.subheader("Transaction Details")
    input_data = {}
    
    # Input fields (adjusted based on typical fraud dataset)
    input_data['amt'] = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    input_data['merchant'] = st.text_input("Merchant", value="merchant_1")
    input_data['category'] = st.selectbox("Category", options=['gas_transport', 'grocery_pos', 'travel', 'other'])
    input_data['gender'] = st.selectbox("Gender", options=['M', 'F'])
    input_data['city'] = st.text_input("City", value="New York")
    input_data['state'] = st.text_input("State", value="NY")
    input_data['job'] = st.text_input("Job", value="Engineer")
    input_data['lat'] = st.number_input("Latitude", value=40.0)
    input_data['long'] = st.number_input("Longitude", value=-74.0)
    input_data['city_pop'] = st.number_input("City Population", min_value=0, value=100000)
    input_data['hour'] = st.number_input("Transaction Hour", min_value=0, max_value=23, value=12)
    input_data['day_of_week'] = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)
    input_data['month'] = st.number_input("Month", min_value=1, max_value=12, value=1)
    
    if st.button("Predict"):
        # Preprocess input
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0  # Handle unseen categories
        
        # Scale numerical features
        numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'hour', 'day_of_week', 'month']
        existing_numerical_cols = [col for col in numerical_cols if col in input_df.columns]
        if existing_numerical_cols:
            input_df[existing_numerical_cols] = scaler.transform(input_df[existing_numerical_cols])
        
        # Ensure input_df has the same columns as training data
        feature_cols = [col for col in hybrid_model.estimators_[0].feature_names_in_]
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)
        
        # Predict
        prediction = hybrid_model.predict(input_df)
        prediction_prob = hybrid_model.predict_proba(input_df)[0]
        
        # Display result
        if prediction[0] == 1:
            st.error(f"Fraudulent Transaction! (Confidence: {prediction_prob[1]:.2%})")
        else:
            st.success(f"Legitimate Transaction (Confidence: {prediction_prob[0]:.2%})")

if __name__ == "__main__":
    main()