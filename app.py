from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("online_food_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "online_food_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "online-food-reorder-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    model = load_model()
    features = list(model.feature_names_in_)

    missing_fields = [f for f in features if f not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{f: data[f] for f in features}])
    raw = model.predict(sample)[0]

    if isinstance(raw, (int, float, np.integer, np.floating)):
        prediction_code = int(raw)
        prediction = "Yes" if prediction_code == 1 else "No"
    else:
        prediction = str(raw)
        prediction_code = 1 if prediction.strip().lower() == "yes" else 0

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
