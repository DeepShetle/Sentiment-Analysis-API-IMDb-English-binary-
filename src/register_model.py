"""
Day 9 - Step 6-8: Register the best model into MLflow Model Registry,
assign 'champion' alias, and verify loading the model from the Registry.
"""

import mlflow
from mlflow import MlflowClient

import os
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# ────────────────────────────────────────────────────────────────
# Step 6: Register model
# Paste the run_id copied from the terminal or MLflow UI
# ────────────────────────────────────────────────────────────────
best_run_id = "039b29d9d5df43b8a9e63bad2fae344c"

result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="sentiment-classifier",
)
print(f"Registered: name={result.name}, version={result.version}")

# ────────────────────────────────────────────────────────────────
# Step 7: Assign 'champion' alias (replaces deprecated 'Production' stage)
# ────────────────────────────────────────────────────────────────
client = MlflowClient()
client.set_registered_model_alias(
    name="sentiment-classifier",
    alias="champion",
    version=result.version,
)
print(f"Assigned alias 'champion' for version {result.version}")

# ────────────────────────────────────────────────────────────────
# Step 8: Verification - load model FROM REGISTRY (not from .pkl file)
# ────────────────────────────────────────────────────────────────
production_model = mlflow.pyfunc.load_model("models:/sentiment-classifier@champion")

sample = ["This movie was absolutely wonderful!", "This was a terrible waste of time."]
predictions = production_model.predict(sample)
print("Sample predictions:", predictions)