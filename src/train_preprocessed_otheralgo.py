# src/train.py

import json
import time
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Load data & tái tạo lại đúng split đã dùng ở Ngày 6-7 ---
df = pd.read_csv("data/processed/imdb_processed.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["review_clean"],
    df["sentiment"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["sentiment"],
)

# reset_index để tránh lệch index khi ghép/lọc DataFrame về sau (thói quen đã thống nhất)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# --- Load lại vectorizer ĐÃ FIT ở Ngày 7 — chỉ transform, tuyệt đối không fit_transform ---
vectorizer_clean = joblib.load("artifacts/vectorizer_clean.pkl")

X_train_vec = vectorizer_clean.transform(X_train)
X_test_vec = vectorizer_clean.transform(X_test)

# assert để bắt lỗi sớm nếu vectorizer không khớp (thói quen đã dùng từ Ngày 6)
assert X_train_vec.shape[1] == len(vectorizer_clean.get_feature_names_out()), \
    "Số chiều feature không khớp với vocabulary của vectorizer đã fit"
print(f"Train shape: {X_train_vec.shape}, Test shape: {X_test_vec.shape}")

#train LinearSVC
print("Training LinearSVC...")
start = time.time()
model_svm = LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
model_svm.fit(X_train_vec, y_train)
train_time_svm = time.time() - start
print(f"  Done in {train_time_svm:.2f}s")

#train random forest
print("Training Random Forest...")
start = time.time()
model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,  # dùng hết CPU cores để bù lại việc RF vốn chậm hơn linear model
)
model_rf.fit(X_train_vec, y_train)
train_time_rf = time.time() - start
print(f"  Done in {train_time_rf:.2f}s")

#tổng hợp, so sánh
results = {}

for name, model, t in [
    ("linear_svc", model_svm, train_time_svm),
    ("random_forest", model_rf, train_time_rf),
]:
    y_pred = model.predict(X_test_vec)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, pos_label="positive"), 4),
        "train_time_seconds": round(t, 2),
    }

results["logistic_regression"] = {
    "accuracy": 0.8953,
    "f1": 0.8969,
    "train_time_seconds": None,  
}

print(json.dumps(results, indent=2))

best_model_name = max(results, key=lambda k: results[k]["f1"])
print(f"\n>>> Model tốt nhất theo F1: {best_model_name}")

#Xử lí predict_proba nếu SVM win
best_model = None

if best_model_name == "linear_svc":
    print("SVM thắng — cần calibrate để có predict_proba đáng tin cậy cho API sau này")
    start = time.time()
    model_svm_calibrated = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=2000),
        cv=5,
    )
    model_svm_calibrated.fit(X_train_vec, y_train)
    calib_time = time.time() - start
    print(f"  Calibration done in {calib_time:.2f}s")

    # Kiểm tra lại proba có hợp lệ không (mỗi hàng phải sum ~ 1.0)
    proba_sample = model_svm_calibrated.predict_proba(X_test_vec[:5])
    assert proba_sample.shape[1] == 2, "Phải có đúng 2 cột xác suất (positive/negative)"
    print("Sample proba:", proba_sample)

    best_model = model_svm_calibrated
    results["linear_svc_calibrated"] = {
        "accuracy": round(accuracy_score(y_test, model_svm_calibrated.predict(X_test_vec)), 4),
        "f1": round(f1_score(y_test, model_svm_calibrated.predict(X_test_vec), pos_label="positive"), 4),
        "train_time_seconds": round(calib_time, 2),
    }
elif best_model_name == "random_forest":
    best_model = model_rf  # RF đã có predict_proba sẵn, không cần xử lý gì thêm
else:
    best_model = None  # tức Logistic Regression đã có sẵn từ Ngày 7, không train lại ở đây

#Note: Hiện tại tôi chưa thực sự hiểu về SVM, bước xử lí này chỉ mang tính thủ tục, hiện tại tôi đang tập trung học các thuật toán phổ biến và hữu dụng hơn thông qua ML specialization

#Lưu kết quả và artifacts

# Lưu bảng so sánh
Path("reports").mkdir(exist_ok=True)
with open("reports/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

# Lưu model tốt nhất — đặt tên phân biệt rõ với model_baseline.pkl / model_clean.pkl
Path("artifacts").mkdir(exist_ok=True)
if best_model is not None:
    joblib.dump(best_model, "artifacts/model_best.pkl")
    print("Đã lưu artifacts/model_best.pkl")
else:
    print("Model tốt nhất là Logistic Regression (Ngày 7) — artifact model_clean.pkl đã có sẵn, không cần lưu lại")
