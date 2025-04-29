# Credit-Card-Fraud-Detection
This Project aims to predict the credit card fraud using the user standard inputs. We have used RandomForest and XGBoost classifier and used them as Hybrid Model using Soft voting ensemble method.


🚀 Credit Card Fraud Detection App

Welcome to the Credit Card Fraud Detection project! This README will guide you through the theory, code structure, model architecture, and how to launch the app. We’ve sprinkled in some emojis to keep things fun and colorful! 🌈

---

📖 Table of Contents  
1. 🔍 Project Overview  
2. 🗂️ File Structure  
3. 🛠️ Installation & Setup  
4. 🔬 Theoretical Background  
5. 🤖 Model Architecture  
6. ⚙️ Code Explanation  
   - • Streamlit App (app.py)  
   - • Training Pipeline (train_model.py / train_randomforest.py)  
7. 🌐 Running the Web App  
8. 📦 Download Pre-trained Model  

---

🔍 Project Overview  
This project detects fraudulent credit-card transactions in real time using a lightweight Streamlit interface. Under the hood, we:  
- 🏗️ Preprocess transaction data (timestamps → time features, encode categoricals, scale numerics, handle imbalance with SMOTE)  
- 🌳 Train two powerful tree-based learners: Random Forest & XGBoost  
- 🤝 Combine them into a soft-voting ensemble (“Hybrid Model”) for robust predictions  
- 🚀 Deploy the model in a Streamlit app for instant fraud/no-fraud feedback  

---

🗂️ File Structure  
fraud-detection-project/  
│  
├─ app.py                    # Streamlit web app for inference  
├─ train_model.py            # Full pipeline: preprocessing, SMOTE, RF+XGB ensemble  
├─ train_randomforest.py     # RF-only training pipeline  
├─ requirements.txt          # Python dependencies  
├─ FraudTrain.csv            # Training data  
├─ FraudTest.csv             # Test data  
├─ hybrid_model.pkl          # Saved VotingClassifier model  
├─ random_forest_model.pkl   # (Optional) Saved RF model  
├─ label_encoders.pkl        # Saved LabelEncoder objects  
└─ README.md                 # This documentation  

---

🛠️ Installation & Setup  
1. Clone the repo  
   git clone https://github.com/your-username/fraud-detection-project.git  
   cd fraud-detection-project  

2. Create & activate virtual environment  
   python3 -m venv venv  
   source venv/bin/activate   # macOS/Linux  
   venv\Scripts\activate      # Windows  

3. Install dependencies  
   pip install -r requirements.txt  

---

🔬 Theoretical Background  
1. SMOTE (Synthetic Minority Over-sampling Technique)  
   - Balances the highly imbalanced fraud dataset by generating synthetic “fraud” examples in feature space.  
2. Random Forest (Bagging)  
   - Aggregates predictions from multiple decision trees trained on bootstrap samples → reduces variance.  
3. XGBoost (Boosting)  
   - Sequentially builds trees that correct previous errors (gradient boosting) → reduces bias.  
4. Soft-Voting Ensemble  
   - Averages the predicted probabilities from RF & XGB, then picks the class with highest mean probability → often outperforms individual models.

---

🤖 Model Architecture  
flowchart:  
Raw Transaction Data → Preprocessing →  
 • Random Forest →  
 • XGBoost →  
Both feed into → VotingClassifier → Fraud Probability & Label  

Preprocessing steps:  
- Timestamp → hour, day_of_week, month  
- Label-encode categorical cols (merchant, category, gender, city, state)  
- Scale numeric cols (amt, geolocation, pop, time features)

Training steps:  
- Split → SMOTE → train RF & XGB → fit VotingClassifier

Inference steps:  
- Identical preprocessing → model.predict_proba() & model.predict()

---

⚙️ Code Explanation  

• Streamlit App (app.py)  
- Load: hybrid_model.pkl, label_encoders.pkl via joblib.load  
- UI Inputs:  
  • Transaction datetime → parsed into hour, day_of_week  
  • Amount, merchant, category, location, gender, etc.  
- Preprocess:  
  • Transform categoricals with saved encoders  
  • Construct a DataFrame in correct feature order  
- Predict:  
  • prob = model.predict_proba(input_df)[0][1] (fraud probability)  
  • label = model.predict(input_df)[0]  
- Display:  
  • Red ⚠️ for fraud, green ✔️ for legitimate, with confidence %

• Training Pipeline  

train_model.py  
1. Load CSVs → concat → drop unused cols  
2. Extract time features from trans_date_trans_time  
3. Label-encode categoricals, save encoders  
4. SMOTE oversampling on training split  
5. Fit RF & XGB on balanced data  
6. Fit VotingClassifier (“Hybrid Model”)  
7. Save hybrid_model.pkl & label_encoders.pkl  

train_randomforest.py  
- Same preprocessing + SMOTE → train only Random Forest → save random_forest_model.pkl  

---

🌐 Running the Web App  
streamlit run app.py  
- Opens at http://localhost:8501  
- Enter transaction details → click Predict → view result instantly!

---

📦 Download Pre-trained Model  
Get the latest hybrid_model.pkl by running the fraud_detection_app.py

---

🎉 You’re all set!  
Feel free to explore, tweak the thresholds, add more features, or swap in new models. Happy hacking! 🚀
