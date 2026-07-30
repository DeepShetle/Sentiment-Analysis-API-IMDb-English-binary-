"""
Ngay 9 - Buoc 6-8: Dang ky model tot nhat vao MLflow Model Registry,
gan alias 'champion', va kiem chung load lai model tu Registry.
"""

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

# ────────────────────────────────────────────────────────────────
# Buoc 6: Dang ky model
# Dan run_id da copy tu terminal (dong "Run_id de dang ky Registry: ...")
# hoac copy tu MLflow UI, trang chi tiet run "custom_preprocessing_logreg"
# ────────────────────────────────────────────────────────────────
best_run_id = "039b29d9d5df43b8a9e63bad2fae344c"

result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="sentiment-classifier",
)
print(f"Da dang ky: name={result.name}, version={result.version}")

# ────────────────────────────────────────────────────────────────
# Buoc 7: Gan alias "champion" (thay cho stage "Production" da deprecated)
# ────────────────────────────────────────────────────────────────
client = MlflowClient()
client.set_registered_model_alias(
    name="sentiment-classifier",
    alias="champion",
    version=result.version,
)
print(f"Da gan alias 'champion' cho version {result.version}")

# ────────────────────────────────────────────────────────────────
# Buoc 8: Kiem chung - load lai model TU REGISTRY (khong phai tu file .pkl)
# ────────────────────────────────────────────────────────────────
production_model = mlflow.pyfunc.load_model("models:/sentiment-classifier@champion")

sample = ["This movie was absolutely wonderful!", "This was a terrible waste of time."]
predictions = production_model.predict(sample)
print("Du doan mau:", predictions)