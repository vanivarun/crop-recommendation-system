# src/train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


def main(data_path='data/crop_data.csv', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    target_col = 'label' if 'label' in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split FIRST — before fitting scaler (prevents data leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)  # only transform, never fit

    # Cross-validation to find best n_estimators
    print("Running cross-validation...")
    best_score = 0
    best_n = 100
    for n in [100, 150, 200]:
        clf_cv = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf_cv, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  n_estimators={n}: CV accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_n = n

    print(f"\nBest n_estimators: {best_n} (CV accuracy: {best_score:.4f})")

    # Train final model
    clf = RandomForestClassifier(n_estimators=best_n, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save confusion matrix and test accuracy for app.py to use
    cm = confusion_matrix(y_test, y_pred)
    joblib.dump(cm,       os.path.join(model_dir, 'confusion_matrix.joblib'))
    joblib.dump(test_acc, os.path.join(model_dir, 'test_accuracy.joblib'))

    # Save model artifacts
    joblib.dump(clf,          os.path.join(model_dir, 'rf_crop_model.joblib'))
    joblib.dump(scaler,       os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le,           os.path.join(model_dir, 'label_encoder.joblib'))
    joblib.dump(list(X.columns), os.path.join(model_dir, 'feature_names.joblib'))

    print(f"\nAll artifacts saved in '{model_dir}/'")


if __name__ == '__main__':
    main()
    