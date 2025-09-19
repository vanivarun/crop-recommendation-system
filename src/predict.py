# src/predict.py
import joblib
import numpy as np
import pandas as pd
import sys, os

def predict(values, model_dir='models'):
    clf = joblib.load(os.path.join(model_dir, 'rf_crop_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    features = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))

    # Convert input to DataFrame with correct column names
    df = pd.DataFrame([values], columns=features)
    arr_scaled = scaler.transform(df)
    pred_enc = clf.predict(arr_scaled)
    pred_label = le.inverse_transform(pred_enc)[0]

    print("Predicted Crop:", pred_label)

if __name__ == '__main__':
    if len(sys.argv) < 8:
        print("Usage: python src/predict.py N P K temperature humidity ph rainfall")
        sys.exit(1)
    values = [float(v) for v in sys.argv[1:]]
    predict(values)
