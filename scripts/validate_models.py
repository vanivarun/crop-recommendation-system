"""Simple model artifacts validator.

Checks that model, scaler, label encoder and feature names load correctly,
and that their shapes are consistent.
"""
import joblib
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent.parent
    model_dir = base / "models"
    report = {"ok": True, "issues": []}

    expected_files = [
        "rf_crop_model.joblib",
        "scaler.joblib",
        "label_encoder.joblib",
        "feature_names.joblib",
    ]

    for fname in expected_files:
        if not (model_dir / fname).exists():
            report["ok"] = False
            report["issues"].append(f"Missing file: {fname}")

    if not report["ok"]:
        print(json.dumps(report, indent=2))
        return

    # Try loading
    model = joblib.load(model_dir / "rf_crop_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")
    features = joblib.load(model_dir / "feature_names.joblib")

    # Basic checks
    try:
        n_features = len(features)
        model_n_in = None
        try:
            model_n_in = model.n_features_in_
        except Exception:
            # estimator may not expose this
            pass

        if model_n_in is not None and model_n_in != n_features:
            report["ok"] = False
            report["issues"].append(f"Model expects {model_n_in} features but feature_names has {n_features}")

        # Check label encoder / classes
        if hasattr(model, "classes_"):
            if len(model.classes_) != len(le.classes_):
                report["ok"] = False
                report["issues"].append("Mismatch between model.classes_ and label encoder classes_")

        # Check scaler compatibility
        try:
            import numpy as np

            dummy = [0.0] * n_features
            _ = scaler.transform([dummy])
        except Exception as e:
            report["ok"] = False
            report["issues"].append(f"Scaler transform failed: {e}")

    except Exception as e:
        report["ok"] = False
        report["issues"].append(str(e))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
