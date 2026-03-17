# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

DATA_DIR = Path("Model files")  # recommended place to keep model files and pickles

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

@st.cache_data
def load_csv_safe(path: Path):
    return pd.read_csv(path)

def preprocess_datetime(df, column='trans_date_trans_time'):
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors='coerce')
    df['hour'] = df[column].dt.hour.fillna(0).astype(int)
    df['day_of_week'] = df[column].dt.dayofweek.fillna(0).astype(int)
    df['month'] = df[column].dt.month.fillna(0).astype(int)
    if column in df.columns:
        df = df.drop(columns=[column])
    return df

def load_pretrained():
    """Load model, scaler, encoders, and feature columns if available"""
    model_path = DATA_DIR / "hybrid_model.pkl"
    scaler_path = DATA_DIR / "scaler.pkl"
    encoders_path = DATA_DIR / "label_encoders.pkl"
    features_path = DATA_DIR / "feature_columns.pkl"

    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = None
    label_encoders = {}
    feature_columns = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    if encoders_path.exists():
        with open(encoders_path, "rb") as f:
            label_encoders = pickle.load(f)
    if features_path.exists():
        with open(features_path, "rb") as f:
            feature_columns = pickle.load(f)

    return dict(
        model=model,
        scaler=scaler,
        label_encoders=label_encoders,
        feature_columns=feature_columns,
    )

def train_and_save(train_path: Path, test_path: Path):
    st.info("Training models — this runs only if no saved model found and CSVs exist.")
    train_df = load_csv_safe(train_path)
    test_df = load_csv_safe(test_path)

    # basic cleaning
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all.fillna(0, inplace=True)

    if 'trans_date_trans_time' in df_all.columns:
        df_all = preprocess_datetime(df_all, 'trans_date_trans_time')

    # Example lists: customize to match your dataset columns
    categorical_cols = [c for c in ['merchant', 'category', 'gender', 'city', 'state', 'job'] if c in df_all.columns]
    numerical_cols = [c for c in ['amt', 'lat', 'long', 'city_pop', 'hour', 'day_of_week', 'month'] if c in df_all.columns]

    # Encode categorical
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str))
        label_encoders[col] = le

    # Scale numerical
    scaler = StandardScaler()
    if numerical_cols:
        df_all[numerical_cols] = scaler.fit_transform(df_all[numerical_cols])

    # split back
    train_len = len(train_df)
    train_processed = df_all.iloc[:train_len].reset_index(drop=True)
    test_processed = df_all.iloc[train_len:].reset_index(drop=True)

    # Target and features
    if 'is_fraud' not in train_processed.columns:
        st.error("Train CSV must contain 'is_fraud' column.")
        return None

    exclude = {'is_fraud', 'first', 'last', 'street', 'dob', 'trans_num'}
    feature_cols = [c for c in train_processed.columns if c not in exclude]

    X_train = train_processed[feature_cols]
    y_train = train_processed['is_fraud']
    X_test = test_processed[feature_cols] if 'is_fraud' in test_processed.columns else None
    y_test = test_processed['is_fraud'] if 'is_fraud' in test_processed.columns else None

    # Models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    hybrid = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')
    hybrid.fit(X_train, y_train)

    # Save artifacts
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "hybrid_model.pkl", "wb") as f:
        pickle.dump(hybrid, f)
    with open(DATA_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(DATA_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open(DATA_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    # Evaluate if test labels exist
    if X_test is not None and y_test is not None:
        preds = hybrid.predict(X_test)
        try:
            probs = hybrid.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, probs)
        except Exception:
            roc = roc_auc_score(y_test, preds)
        st.subheader("Evaluation on test set")
        st.text(classification_report(y_test, preds))
        st.write("ROC AUC (approx):", roc)
    st.success("Training completed and artifacts saved to 'Model files/'")
    return dict(model=hybrid, scaler=scaler, label_encoders=label_encoders, feature_columns=feature_cols)

def safe_encode_input(df, label_encoders):
    df = df.copy()
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df.at[0, col]
            try:
                df[col] = le.transform([str(val)])[0]
            except Exception:
                # unseen category: map to 0 and warn
                df[col] = 0
                st.warning(f"Input value for '{col}' was unseen in training; encoded as 0.")
    return df

def main():
    st.title("Credit Card Fraud Detection")
    st.write("Enter transaction details to predict if it's fraudulent.")

    model_bundle = load_pretrained()
    # if not found, try to train if CSVs exist in repo root
    if model_bundle is None:
        train_path = Path("FraudTrain.csv")
        test_path = Path("FraudTest.csv")
        if train_path.exists() and test_path.exists():
            model_bundle = train_and_save(train_path, test_path)
        else:
            st.warning("No saved model found and training CSVs not present in repo.")
            st.info("To deploy quickly: train locally and upload files into 'Model files/' or upload 'hybrid_model.pkl', 'scaler.pkl', 'label_encoders.pkl', 'feature_columns.pkl'.")
            st.stop()

    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    label_encoders = model_bundle.get("label_encoders", {})
    feature_cols = model_bundle.get("feature_columns")

    # User input
    st.subheader("Transaction Details")
    with st.form("tx_form"):
        amt = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        merchant = st.text_input("Merchant", value="merchant_1")
        category = st.selectbox("Category", options=['gas_transport', 'grocery_pos', 'travel', 'other'])
        gender = st.selectbox("Gender", options=['M', 'F'])
        city = st.text_input("City", value="New York")
        state = st.text_input("State", value="NY")
        job = st.text_input("Job", value="Engineer")
        lat = st.number_input("Latitude", value=40.0, format="%.6f")
        long = st.number_input("Longitude", value=-74.0, format="%.6f")
        city_pop = st.number_input("City Population", min_value=0, value=100000)
        hour = st.slider("Transaction Hour", 0, 23, 12)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
        month = st.slider("Month", 1, 12, 1)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            "amt": amt,
            "merchant": merchant,
            "category": category,
            "gender": gender,
            "city": city,
            "state": state,
            "job": job,
            "lat": lat,
            "long": long,
            "city_pop": city_pop,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
        }
        input_df = pd.DataFrame([input_dict])

        # encode categorical using saved encoders (unseen -> 0)
        input_df = safe_encode_input(input_df, label_encoders)

        # scale numeric
        num_cols = [c for c in ['amt', 'lat', 'long', 'city_pop', 'hour', 'day_of_week', 'month'] if c in input_df.columns]
        if scaler is not None and num_cols:
            try:
                input_df[num_cols] = scaler.transform(input_df[num_cols])
            except Exception as e:
                st.warning("Scaler transform failed: " + str(e))

        # align columns
        if feature_cols is None:
            # fallback: use model's first estimator feature names if available
            try:
                feature_cols = list(model.estimators_[0].feature_names_in_)
            except Exception:
                feature_cols = list(input_df.columns)  # last resort

        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        try:
            pred = model.predict(input_df)
            proba = model.predict_proba(input_df)[0]
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            return

        if pred[0] == 1:
            st.error(f"Fraudulent Transaction! (Confidence: {proba[1]:.2%})")
        else:
            st.success(f"Legitimate Transaction (Confidence: {proba[0]:.2%})")

if __name__ == "__main__":
    main()
