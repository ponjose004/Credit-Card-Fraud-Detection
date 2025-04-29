import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime

# Load datasets
train_df = pd.read_csv('FraudTrain.csv')
test_df = pd.read_csv('FraudTest.csv')

# Combine datasets for consistent preprocessing
df = pd.concat([train_df, test_df], ignore_index=True)

# Drop irrelevant columns (including Unnamed: 0)
drop_cols = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'job', 'dob', 'trans_num', 'unix_time']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Convert trans_date_trans_time to datetime and extract features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.hour
df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
df = df.drop(columns=['trans_date_trans_time'])

# Load label encoders (created by train_random_forest.py)
label_encoders = joblib.load('label_encoders.pkl')

# Encode categorical variables using existing encoders
cat_cols = ['merchant', 'category', 'gender', 'city', 'state']
for col in cat_cols:
    df[col] = label_encoders[col].transform(df[col])

# Define features and target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize and train XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_res, y_train_res)

# Save the XGBoost model
joblib.dump(xgb_model, 'xgboost_model.pkl')

print("XGBoost model training complete. Model saved.")