import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import uuid
from datetime import datetime

# Load datasets
train_df = pd.read_csv('FraudTrain.csv')
test_df = pd.read_csv('FraudTest.csv')

# Combine datasets for consistent preprocessing
df = pd.concat([train_df, test_df], ignore_index=True)

# Drop irrelevant columns
drop_cols = ['cc_num', 'first', 'last', 'street', 'job', 'dob', 'trans_num', 'unix_time']
df = df.drop(columns=drop_cols)

# Convert trans_date_trans_time to datetime and extract features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.hour
df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
df = df.drop(columns=['trans_date_trans_time'])

# Encode categorical variables
cat_cols = ['merchant', 'category', 'gender', 'city', 'state']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders for Streamlit app
joblib.dump(label_encoders, 'label_encoders.pkl')

# Define features and target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train individual models
rf_model.fit(X_train_res, y_train_res)
xgb_model.fit(X_train_res, y_train_res)

# Create hybrid model (Voting Classifier)
hybrid_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('xgb', xgb_model)
], voting='soft')

# Train hybrid model
hybrid_model.fit(X_train_res, y_train_res)

# Save the hybrid model
joblib.dump(hybrid_model, 'hybrid_model.pkl')

print("Model training complete. Hybrid model and label encoders saved.")