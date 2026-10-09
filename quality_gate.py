import json
import sys

THRESHOLD = 0.80

with open("metrics.json") as file:
    metrics = json.load(file)

acc = metrics["accuracy"]
print(f"Accuracy: {acc} | Required: {THRESHOLD}")

if acc < THRESHOLD:
    print("Quality gate FAILED")
    sys.exit(1)
print("Quality gate PASSED")
