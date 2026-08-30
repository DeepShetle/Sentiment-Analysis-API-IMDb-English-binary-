
import joblib
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
MAX_FEATURES = 10000

import os
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("sentiment-analysis")


# ────────────────────────────────────────────────────────────────
# 1. Load pre-trained artifacts (Baseline + Logistic Regression with preprocessing)
#    Update path to match the actual file name in artifacts/
# ────────────────────────────────────────────────────────────────
vectorizer_baseline = joblib.load("artifacts/vectorizer_baseline.pkl")
model_baseline       = joblib.load("artifacts/model_baseline.pkl")

vectorizer_clean   = joblib.load("artifacts/vectorizer_clean.pkl")
model_logreg_clean = joblib.load("artifacts/model_logreg_clean.pkl")


# ────────────────────────────────────────────────────────────────
# 2. Combine vectorizer + model into a Pipeline (packaging only, no refitting)
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
# 3. Log 4 runs
# ────────────────────────────────────────────────────────────────

# Run 1: Baseline - review_baseline, lowercase only
# Using 'with' block context manager starts and ends the run automatically
with mlflow.start_run(run_name="baseline_logreg"):  # A run is the smallest unit in MLflow representing an experiment/training session
    mlflow.log_param("preprocessing", "lowercase only")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)  # log_param() for logging training configurations
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8935)  # log_metric() for logging results
    mlflow.log_metric("f1", 0.8951)
    mlflow.log_metric("train_time_seconds", 0.60)
    mlflow.sklearn.log_model(pipeline_baseline, "model") # save the model

# Run 2: Custom preprocessing - review_clean
# -> this is the model that will be registered in the Registry (best accuracy and f1)
with mlflow.start_run(run_name="custom_preprocessing_logreg"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8953)
    mlflow.log_metric("f1", 0.8969)
    mlflow.log_metric("train_time_seconds", 0.55)
    mlflow.sklearn.log_model(pipeline_logreg_clean, "model")
    print("Run_id for Registry:", mlflow.active_run().info.run_id)

# Run 3: SVM - artifacts not saved, metrics only
with mlflow.start_run(run_name="svm_uncalibrated"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "LinearSVC (not calibrated)")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8923)
    mlflow.log_metric("f1", 0.8934)
    mlflow.log_metric("train_time_seconds", 1.86)
    mlflow.set_tag("artifact_status", "not_saved_metrics_only")

# Run 4: Random Forest - artifacts not saved, metrics only
with mlflow.start_run(run_name="random_forest"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8405)
    mlflow.log_metric("f1", 0.8404)
    mlflow.log_metric("train_time_seconds", 17.98)
    mlflow.set_tag("artifact_status", "not_saved_metrics_only")

print("Finished logging 4 runs. Open http://localhost:5000 to view.")