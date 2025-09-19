 # src/train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main(data_path='data/crop_data.csv', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    target_col = 'label' if 'label' in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    joblib.dump(clf, os.path.join(model_dir, 'rf_crop_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))
    joblib.dump(list(X.columns), os.path.join(model_dir, 'feature_names.joblib'))
    print("Model and artifacts saved in", model_dir)

if __name__ == '__main__':
    main()

