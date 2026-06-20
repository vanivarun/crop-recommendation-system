import unittest
from pathlib import Path

from src.predict import build_input_dataframe, load_artifacts, predict


class TestPredictEdgeCases(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_dir = self.base_dir / "models"

    def test_missing_model_dir_raises(self):
        missing = self.base_dir / "nonexistent_models"
        with self.assertRaises(FileNotFoundError):
            load_artifacts(missing)

    def test_invalid_input_length_raises(self):
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        values = [1, 2, 3]
        with self.assertRaises(ValueError):
            build_input_dataframe(values, features)

    def test_nonfinite_value_returns_error(self):
        values = [90.0, 42.0, 43.0, float("nan"), 82.0, 6.5, 202.9]
        result = predict(values, model_dir=str(self.model_dir))
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
