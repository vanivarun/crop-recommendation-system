import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# Load trained model and preprocessing tools
model = joblib.load("models/rf_crop_model.joblib")  
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")
feature_names = joblib.load("models/feature_names.joblib")
# App title
st.title("🌾 Crop Recommendation System")
st.write("Enter soil and weather conditions to get the best crop suggestion.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.number_input("Temperature (°C)", value=20.5)
humidity = st.number_input("Humidity (%)", value=80.0)
ph = st.number_input("pH value", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=200.0)

# Predict button
if st.button("Recommend Crop"):
    # Prepare input features
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    crop = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🌱 Recommended Crop: **{crop}**")

# --- Model Performance Section ---
st.subheader("📊 Model Performance")

# Load dataset (to calculate metrics again)
try:
    data = pd.read_csv("data/crop_data.csv")
    X = data.drop("label", axis=1)
    y = data["label"]

    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Accuracy
    acc = accuracy_score(label_encoder.transform(y), y_pred)
    st.write(f"✅ Model Accuracy: **{acc*100:.2f}%**")

    # Confusion Matrix
    cm = confusion_matrix(label_encoder.transform(y), y_pred)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

except Exception as e:
    st.warning("⚠️ Could not load dataset for metrics/visuals. Please ensure `data/crop_data.csv` exists.")
    st.text(str(e))
