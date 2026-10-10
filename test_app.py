import json
import unittest

import pandas as pd

from app import app, load_model


class TestPredictionApplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.features = list(cls.model.feature_names_in_)
        df = pd.read_csv("onlinefoods.csv")
        cls.samples = json.loads(
            df[cls.features].head(5).to_json(orient="records")
        )

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction(self):
        response = self.client.post("/predict", json=self.samples[0])
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["prediction"], ["Yes", "No"])

    def test_multiple_predictions_are_valid(self):
        for sample in self.samples:
            response = self.client.post("/predict", json=sample)
            self.assertEqual(response.status_code, 200)
            self.assertIn(response.get_json()["prediction"], ["Maybe"])

    def test_missing_field_validation(self):
        first = self.features[0]
        response = self.client.post(
            "/predict", json={first: self.samples[0][first]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
