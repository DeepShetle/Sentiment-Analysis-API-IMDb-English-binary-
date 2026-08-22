"""
Bootstrap script: Log model vao MLflow + Register + Gan alias "champion".
Chay BEN TRONG Docker container khi setup lan dau.

Gop logic tu:
  - src/log_experiments.py  (log runs)
  - src/register_model.py   (register + alias)

Khac biet chinh so voi 2 file goc:
  - tracking_uri lay tu env var (http://mlflow:5000), khong hardcode localhost
  - run_id lay DONG tu mlflow.active_run(), khong hardcode
  - Co verification step cuoi cung
  - Exit code 0 (thanh cong) / 1 (that bai) de Docker biet ket qua
"""

import os
import sys

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline


# ────────────────────────────────────────────────────────────────
# 1. Setup MLflow — dung hostname Docker, KHONG dung localhost
# ────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]  # http://mlflow:5000
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("sentiment-analysis")

RANDOM_STATE = 42
MAX_FEATURES = 10000
MODEL_NAME = os.environ.get("MODEL_NAME", "sentiment-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

print(f"[bootstrap] MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"[bootstrap] Model: {MODEL_NAME}@{MODEL_ALIAS}")


# ────────────────────────────────────────────────────────────────
# 2. Load .pkl artifacts da train san (mount volume tu host)
# ────────────────────────────────────────────────────────────────
print("[bootstrap] Loading .pkl artifacts...")
vectorizer_baseline = joblib.load("artifacts/vectorizer_baseline.pkl")
model_baseline = joblib.load("artifacts/model_baseline.pkl")

vectorizer_clean = joblib.load("artifacts/vectorizer_clean.pkl")
model_logreg_clean = joblib.load("artifacts/model_logreg_clean.pkl")


# ────────────────────────────────────────────────────────────────
# 3. Gop vectorizer + model thanh Pipeline (chi dong goi, khong fit lai)
# ────────────────────────────────────────────────────────────────
pipeline_baseline = Pipeline([
    ("tfidf", vectorizer_baseline),
    ("classifier", model_baseline),
])

pipeline_logreg_clean = Pipeline([
    ("tfidf", vectorizer_clean),
    ("classifier", model_logreg_clean),
])


# ────────────────────────────────────────────────────────────────
# 4. Log 2 run co model vao MLflow
# ────────────────────────────────────────────────────────────────
print("[bootstrap] Logging runs to MLflow...")

# Run 1: Baseline (Ngay 6) — lowercase only
with mlflow.start_run(run_name="baseline_logreg"):
    mlflow.log_param("preprocessing", "lowercase only")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8935)
    mlflow.log_metric("f1", 0.8951)
    mlflow.log_metric("train_time_seconds", 0.60)
    mlflow.sklearn.log_model(pipeline_baseline, "model")

# Run 2: Custom preprocessing (Ngay 7) — model tot nhat
with mlflow.start_run(run_name="custom_preprocessing_logreg") as best_run:
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8953)
    mlflow.log_metric("f1", 0.8969)
    mlflow.log_metric("train_time_seconds", 0.55)
    mlflow.sklearn.log_model(pipeline_logreg_clean, "model")
    best_run_id = best_run.info.run_id  # lay DONG, khong hardcode

print(f"[bootstrap] Best run_id: {best_run_id}")


# ────────────────────────────────────────────────────────────────
# 5. Register model + gan alias "champion"
# ────────────────────────────────────────────────────────────────
print(f"[bootstrap] Registering model '{MODEL_NAME}'...")
result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME,
)
print(f"[bootstrap] Registered: name={result.name}, version={result.version}")

client = MlflowClient()
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=MODEL_ALIAS,
    version=result.version,
)
print(f"[bootstrap] Alias '{MODEL_ALIAS}' -> version {result.version}")


# ────────────────────────────────────────────────────────────────
# 6. Verification — load lai tu Registry, predict thu
# ────────────────────────────────────────────────────────────────
print(f"[bootstrap] Verifying: loading models:/{MODEL_NAME}@{MODEL_ALIAS}...")
try:
    loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    sample = ["This movie was absolutely wonderful!", "This was a terrible waste of time."]
    predictions = loaded_model.predict(sample)
    print(f"[bootstrap] Verification PASSED — predictions: {predictions}")
    sys.exit(0)
except Exception as e:
    print(f"[bootstrap] Verification FAILED: {e}")
    sys.exit(1)
