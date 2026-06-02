# src/predict.py
import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd


def load_artifacts(model_dir: Path):
    clf = joblib.load(model_dir / "rf_crop_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")
    features = joblib.load(model_dir / "feature_names.joblib")
    if not isinstance(features, (list, tuple)):
        features = list(features)
    return clf, scaler, le, list(features)


def build_input_dataframe(values: Sequence[float], features: Sequence[str]) -> pd.DataFrame:
    if len(values) != len(features):
        raise ValueError(f"Expected {len(features)} input values, got {len(values)}.")
    return pd.DataFrame([values], columns=list(features))


def predict(values: Sequence[float], model_dir: str = "models") -> dict:
    base_dir = Path(__file__).resolve().parent.parent
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_absolute():
        model_dir_path = base_dir / model_dir_path

    clf, scaler, le, features = load_artifacts(model_dir_path)
    df_input = build_input_dataframe(values, features)
    arr_scaled = scaler.transform(df_input)

    result = {"predicted": None, "probabilities": None}
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(arr_scaled)[0]
        class_names = le.inverse_transform(clf.classes_)
        proba_pairs = sorted(
            zip(class_names.tolist(), probs.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        result["predicted"] = proba_pairs[0][0]
        result["probabilities"] = proba_pairs
    else:
        pred_enc = clf.predict(arr_scaled)
        result["predicted"] = le.inverse_transform(pred_enc)[0]

    print(json.dumps(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the best crop recommendation from soil and weather inputs."
    )
    parser.add_argument(
        "values",
        nargs=7,
        type=float,
        metavar=("N", "P", "K", "temperature", "humidity", "ph", "rainfall"),
        help="Seven numeric inputs for the crop recommendation model.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing the model artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.values, model_dir=args.model_dir)
