Model artifacts
===============

This folder contains the trained model and preprocessing artifacts.

- `rf_crop_model.joblib` — trained classifier (RandomForest)
- `scaler.joblib` — StandardScaler instance used to scale inputs
- `label_encoder.joblib` — LabelEncoder used for target labels
- `feature_names.joblib` — list of feature column names in order

If you retrain the model with `python src/train_model.py`, these files will be overwritten.

Use `python scripts/validate_models.py` to run a quick consistency check.
