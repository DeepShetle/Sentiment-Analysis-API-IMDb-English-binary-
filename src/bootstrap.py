"""
Bootstrap script: Log model into MLflow + Register + Assign 'champion' alias.
Runs INSIDE the Docker container during initial setup.

Combined logic from:
  - src/log_experiments.py (log runs)
  - src/register_model.py (register + alias)

Main differences from original files:
  - tracking_uri is retrieved from env var (http://mlflow:5000), not hardcoded to localhost
  - run_id is fetched DYNAMICALLY from mlflow.active_run(), not hardcoded
  - Includes a final verification step
  - Returns exit code 0 (success) / 1 (failure) so Docker knows the result
"""

import os
import sys

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline


# ────────────────────────────────────────────────────────────────
# 1. Setup MLflow — use Docker hostname, DO NOT use localhost
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
# 2. Load pre-trained .pkl artifacts (mounted volume from host)
# ────────────────────────────────────────────────────────────────
print("[bootstrap] Loading .pkl artifacts...")
vectorizer_baseline = joblib.load("artifacts/vectorizer_baseline.pkl")
model_baseline = joblib.load("artifacts/model_baseline.pkl")

vectorizer_clean = joblib.load("artifacts/vectorizer_clean.pkl")
model_logreg_clean = joblib.load("artifacts/model_logreg_clean.pkl")


# ────────────────────────────────────────────────────────────────
# 3. Combine vectorizer + model into a Pipeline (packaging only, no refitting)
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
# 4. Log 2 runs with models to MLflow
# ────────────────────────────────────────────────────────────────
print("[bootstrap] Logging runs to MLflow...")

# Run 1: Baseline — lowercase only
with mlflow.start_run(run_name="baseline_logreg"):
    mlflow.log_param("preprocessing", "lowercase only")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8935)
    mlflow.log_metric("f1", 0.8951)
    mlflow.log_metric("train_time_seconds", 0.60)
    mlflow.sklearn.log_model(pipeline_baseline, "model")

# Run 2: Custom preprocessing — best model
with mlflow.start_run(run_name="custom_preprocessing_logreg") as best_run:
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8953)
    mlflow.log_metric("f1", 0.8969)
    mlflow.log_metric("train_time_seconds", 0.55)
    mlflow.sklearn.log_model(pipeline_logreg_clean, "model")
    best_run_id = best_run.info.run_id  # fetch DYNAMICALLY, do not hardcode

# Run 3: Linear SVC (Hardcoded for UI display)
with mlflow.start_run(run_name="custom_preprocessing_svm"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "LinearSVC")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8923)
    mlflow.log_metric("f1", 0.8934)
    mlflow.log_metric("train_time_seconds", 2.28)
    # Do not log_model because this model underperforms Logistic Regression, only for UI display

# Run 4: Random Forest (Hardcoded for UI display)
with mlflow.start_run(run_name="custom_preprocessing_rf"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8405)
    mlflow.log_metric("f1", 0.8404)
    mlflow.log_metric("train_time_seconds", 21.81)
    # Do not log_model because this model underperforms Logistic Regression, only for UI display

print(f"[bootstrap] Best run_id: {best_run_id}")


# ────────────────────────────────────────────────────────────────
# 5. Register model + assign alias "champion"
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
# 6. Verification — reload from Registry and test prediction
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
