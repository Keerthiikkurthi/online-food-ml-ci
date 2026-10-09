import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("onlinefoods.csv")
data = data.loc[:, ~data.columns.str.startswith("Unnamed")]

X = data.drop(columns=["Output"])
y = data["Output"]

cat_cols = X.select_dtypes(include="object").columns
prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough",
)
model = Pipeline([
    ("prep", prep),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "training_records": len(X_train),
    "testing_records": len(X_test),
}

joblib.dump(model, "online_food_model.pkl")
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

out = X_test.copy()
out["Actual"] = y_test
out["Predicted"] = pred
out.to_csv("test_predictions.csv", index=False)

print(metrics)
