import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
MAX_FEATURES = 10000

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentiment-analysis")


# ────────────────────────────────────────────────────────────────
# 1. Load artifact da fit san (chi con Ngay 6 + Ngay 7, Logistic thang)
#    Sua duong dan cho khop ten file that trong models/
# ────────────────────────────────────────────────────────────────
vectorizer_baseline = joblib.load("models/vectorizer_baseline.pkl")
model_baseline       = joblib.load("models/model_baseline.pkl")

vectorizer_clean   = joblib.load("models/vectorizer_clean.pkl")
model_logreg_clean = joblib.load("models/model_logreg_clean.pkl")


# ────────────────────────────────────────────────────────────────
# 2. Gop vectorizer + model thanh Pipeline (chi dong goi, khong fit lai)
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
# 3. Log 4 run
# ────────────────────────────────────────────────────────────────

# Run 1: Baseline (Ngay 6) - review_baseline, chi lowercase
with mlflow.start_run(run_name="baseline_logreg"):
    mlflow.log_param("preprocessing", "lowercase only")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8935)
    mlflow.log_metric("f1", 0.8951)
    mlflow.sklearn.log_model(pipeline_baseline, "model")

# Run 2: Custom preprocessing (Ngay 7) - review_clean
# -> day la model se duoc dang ky vao Registry (thang ca accuracy va f1)
with mlflow.start_run(run_name="custom_preprocessing_logreg"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8953)
    mlflow.log_metric("f1", 0.8969)
    mlflow.set_tag("train_time_note", "not measured for logistic_regression")
    mlflow.sklearn.log_model(pipeline_logreg_clean, "model")
    print("Run_id de dang ky Registry:", mlflow.active_run().info.run_id)

# Run 3: SVM (Ngay 8) - khong con artifact .pkl, chi con so lieu
with mlflow.start_run(run_name="svm_calibrated"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "LinearSVC (calibrated)")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8923)
    mlflow.log_metric("f1", 0.8934)
    mlflow.log_metric("train_time_seconds", 1.86)
    mlflow.set_tag("artifact_status", "not_saved_metrics_only")

# Run 4: Random Forest (Ngay 8) - tuong tu, khong con artifact
with mlflow.start_run(run_name="random_forest"):
    mlflow.log_param("preprocessing", "teencode + emoji + html cleaning")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", 0.8405)
    mlflow.log_metric("f1", 0.8404)
    mlflow.log_metric("train_time_seconds", 17.98)
    mlflow.set_tag("artifact_status", "not_saved_metrics_only")

print("Da log xong 4 run. Mo http://localhost:5000 de xem.")