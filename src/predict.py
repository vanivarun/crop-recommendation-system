# src/predict.py
import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

# Cache loaded models at module level (avoids reloading on repeated calls)
_cache = {}

# Input validation ranges (based on training data)
RANGES = {
    'N':           (0,    140,  'Nitrogen (N)'),
    'P':           (5,    145,  'Phosphorus (P)'),
    'K':           (5,    205,  'Potassium (K)'),
    'temperature': (8.0,  44.0, 'Temperature (°C)'),
    'humidity':    (14.0, 100.0,'Humidity (%)'),
    'ph':          (3.5,  9.95, 'pH'),
    'rainfall':    (20.0, 299.0,'Rainfall (mm)'),
}


def load_artifacts(model_dir: Path):
    """Load model artifacts from disk, with module-level caching."""
    key = str(model_dir)
    if key not in _cache:
        required = [
            'rf_crop_model.joblib',
            'scaler.joblib',
            'label_encoder.joblib',
            'feature_names.joblib',
        ]
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        for fname in required:
            if not (model_dir / fname).exists():
                raise FileNotFoundError(f"Required artifact missing: {model_dir / fname}")

        clf      = joblib.load(model_dir / 'rf_crop_model.joblib')
        scaler   = joblib.load(model_dir / 'scaler.joblib')
        le       = joblib.load(model_dir / 'label_encoder.joblib')
        features = joblib.load(model_dir / 'feature_names.joblib')
        if not isinstance(features, (list, tuple)):
            features = list(features)

        _cache[key] = (clf, scaler, le, list(features))

    return _cache[key]


def validate_inputs(values: Sequence[float]) -> list[str]:
    """Return a list of error messages for out-of-range inputs."""
    keys = list(RANGES.keys())
    errors = []
    for i, val in enumerate(values):
        key = keys[i]
        lo, hi, label = RANGES[key]
        if not (lo <= val <= hi):
            errors.append(f"{label}: {val} is out of valid range [{lo}, {hi}]")
    return errors


def build_input_dataframe(values: Sequence[float], features: Sequence[str]) -> pd.DataFrame:
    """Build a single-row DataFrame from input values."""
    if len(values) != len(features):
        raise ValueError(f"Expected {len(features)} values, got {len(values)}.")
    arr = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            raise ValueError(f"All inputs must be numeric. Could not convert: {v}")
        if not (pd.notna(fv) and (fv == fv)):
            raise ValueError(f"Input contains non-finite value: {v}")
        arr.append(fv)
    return pd.DataFrame([arr], columns=list(features))


def predict(values: Sequence[float], model_dir: str = 'models', top_n: int = 3) -> dict:
    """
    Predict best crop(s) for given soil/weather conditions.

    Args:
        values:    [N, P, K, temperature, humidity, ph, rainfall]
        model_dir: path to model artifacts directory
        top_n:     number of top recommendations to return

    Returns:
        dict with keys 'predicted' (str) and 'probabilities' (list of [crop, pct] pairs)
    """
    base_dir = Path(__file__).resolve().parent.parent
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_absolute():
        model_dir_path = base_dir / model_dir_path

    try:
        clf, scaler, le, features = load_artifacts(model_dir_path)

        errors = validate_inputs(values)
        if errors:
            return {'error': errors}

        df_input   = build_input_dataframe(values, features)
        arr_scaled = scaler.transform(df_input)

        result = {'predicted': None, 'probabilities': None}

        if hasattr(clf, 'predict_proba'):
            probs        = clf.predict_proba(arr_scaled)[0]
            class_names  = le.inverse_transform(clf.classes_)
            proba_pairs  = sorted(
                zip(class_names.tolist(), (probs * 100).tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            result['predicted']     = proba_pairs[0][0]
            result['probabilities'] = proba_pairs[:top_n]
        else:
            pred_enc          = clf.predict(arr_scaled)
            result['predicted'] = le.inverse_transform(pred_enc)[0]

        return result

    except Exception as exc:
        err_msg = str(exc)
        print(json.dumps({'error': err_msg}))
        print(f"Error: {err_msg}", file=sys.stderr)
        return {'error': err_msg}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Predict the best crop from soil and weather inputs.'
    )
    parser.add_argument(
        'values',
        nargs=7,
        type=float,
        metavar=('N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'),
        help='Seven numeric inputs for the crop recommendation model.',
    )
    parser.add_argument(
        '--model-dir',
        default='models',
        help='Directory containing the model artifacts.',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top crop recommendations to return.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    result = predict(args.values, model_dir=args.model_dir, top_n=args.top_n)

    if 'error' in result:
        print('\n⚠️  Errors:')
        errs = result['error']
        if isinstance(errs, list):
            for e in errs:
                print(f'  - {e}')
        else:
            print(f'  {errs}')
        sys.exit(1)

    print(f"\n🌾 Top {args.top_n} Crop Recommendations:")
    for i, (crop, conf) in enumerate(result['probabilities'], 1):
        print(f"  {i}. {crop:<15} ({conf:.1f}% confidence)")