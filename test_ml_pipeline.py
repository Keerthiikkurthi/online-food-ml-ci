import json
import os
import unittest
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):
    def test_outputs_exist(self):
        for f in ["online_food_model.pkl", "metrics.json", "test_predictions.csv"]:
            self.assertTrue(os.path.exists(f), f"{f} missing")

    def test_metrics_valid(self):
        with open("metrics.json") as file:
            m = json.load(file)
        self.assertGreaterEqual(m["accuracy"], 0.80)
        self.assertGreater(m["training_records"], 0)

    def test_model_predicts(self):
        model = joblib.load("online_food_model.pkl")
        sample = pd.read_csv("test_predictions.csv").drop(columns=["Actual", "Predicted"]).head(5)
        preds = model.predict(sample)
        self.assertEqual(len(preds), 5)


if __name__ == "__main__":
    unittest.main()
