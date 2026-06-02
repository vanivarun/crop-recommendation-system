import unittest
from pathlib import Path

from src.predict import build_input_dataframe, load_artifacts, predict


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_dir = self.base_dir / "models"

    def test_load_artifacts(self):
        clf, scaler, le, features = load_artifacts(self.model_dir)
        self.assertTrue(hasattr(clf, "predict"))
        self.assertTrue(hasattr(scaler, "transform"))
        self.assertTrue(hasattr(le, "inverse_transform"))
        self.assertEqual(len(features), 7)

    def test_build_input_dataframe(self):
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        values = [90.0, 42.0, 43.0, 20.5, 80.0, 6.5, 200.0]
        df = build_input_dataframe(values, features)
        self.assertEqual(df.shape, (1, 7))
        self.assertListEqual(df.columns.tolist(), features)

    def test_predict_returns_expected_structure(self):
        values = [90.0, 42.0, 43.0, 20.87974371, 82.00274423, 6.502985292000001, 202.9355362]
        result = predict(values, model_dir=str(self.model_dir))
        self.assertIsInstance(result, dict)
        self.assertIn("predicted", result)
        self.assertIsInstance(result["predicted"], str)
        self.assertIn("probabilities", result)
        self.assertTrue(result["probabilities"] is None or isinstance(result["probabilities"], list))
        if result["probabilities"] is not None:
            self.assertGreater(len(result["probabilities"]), 0)


if __name__ == "__main__":
    unittest.main()
