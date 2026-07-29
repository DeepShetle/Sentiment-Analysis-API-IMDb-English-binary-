from load_data import load_imdb_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import RANDOM_STATE, TEST_SIZE, MAX_FEATURES

# --- Load data ---
df = load_imdb_data("data/processed/imdb_processed.csv") 

# Assert sớm để bắt lỗi ngay nếu file bị thay đổi ngoài ý muốn giữa các ngày
assert "review_clean" in df.columns, "Thiếu cột review_clean — kiểm tra lại imdb_processed.csv"
assert df["sentiment"].isin(["positive", "negative"]).all(), "Có nhãn lạ ngoài positive/negative"

# --- Split — PHẢI giống hệt Ngày 6 (cùng random_state, cùng cột stratify) ---
X_train, X_test, y_train, y_test = train_test_split(
    df["review_clean"],
    df["sentiment"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["sentiment"],
)

# --- Kiểm tra bắt buộc sau khi split (không bỏ qua) ---

# 1. Kiểm tra kích thước 2 tập
print(f"Train: {len(X_train)} dòng, Test: {len(X_test)} dòng")
assert len(X_train) + len(X_test) == len(df), "Tổng train+test phải bằng tổng dataset"

# 2. Kiểm tra tỷ lệ class có thực sự giữ nguyên sau stratify không
print("Tỷ lệ class trong train:")
print(y_train.value_counts(normalize=True))
print("\nTỷ lệ class trong test:")
print(y_test.value_counts(normalize=True))
print("\nTỷ lệ class trong toàn bộ dataset (để đối chiếu):")
print(df["sentiment"].value_counts(normalize=True))

# 3. Kiểm tra không có leakage giữa train/test (index không trùng nhau)
assert set(X_train.index).isdisjoint(set(X_test.index)), "Train và test không được trùng index"

#4. Reset index

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# --- Vectorize — vectorizer MỚI, fit riêng trên review_clean ---
vectorizer_clean = TfidfVectorizer(max_features=MAX_FEATURES, lowercase = False)
X_train_vec = vectorizer_clean.fit_transform(X_train)
X_test_vec = vectorizer_clean.transform(X_test)  # chỉ transform, không fit lại trên test

# --- Train model — cấu hình giống hệt Ngày 6 ---
# Baseline hội tụ ở max_iter = 1000 nên ko cần check lại nữa, train luôn
model_clean = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model_clean.fit(X_train_vec, y_train)

#---Đánh giá - accuracy + F1 ---

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
#import các chỉ số đo lường

y_pred = model_clean.predict(X_test_vec)     #Dự đoán nhãn cho tập X_test_vec

# Kiểm tra bắt buộc: số lượng dự đoán phải khớp số lượng mẫu test
assert len(y_pred) == len(y_test), "Số dự đoán phải bằng số mẫu test"

acc_clean = accuracy_score(y_test, y_pred)
f1_clean = f1_score(y_test, y_pred, pos_label="positive")

print(f"Clean — Accuracy: {acc_clean:.4f}, F1: {f1_clean:.4f}")
print("\nClassification report chi tiết:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=["negative", "positive"]))

import json
import os

# --- Điền số thật từ Ngày 6, KHÔNG phải giá trị ví dụ ---
acc_baseline = 0.8935   # <-- thay bằng số thật em đã ghi lại ở Ngày 6
f1_baseline = 0.8951    # <-- thay bằng số thật em đã ghi lại ở Ngày 6

comparison = {
    "baseline": {
        "preprocessing": "lowercase only",
        "accuracy": round(acc_baseline, 4),
        "f1": round(f1_baseline, 4),
    },
    "custom_preprocessing": {
        "preprocessing": "teencode + emoji + html cleaning",
        "accuracy": round(acc_clean, 4),
        "f1": round(f1_clean, 4),
    },
    "improvement": {
        "accuracy_delta": round(acc_clean - acc_baseline, 4),
        "f1_delta": round(f1_clean - f1_baseline, 4),
    },
}

print(json.dumps(comparison, indent=2))

os.makedirs("reports", exist_ok=True)
with open("reports/ablation_results.json", "w") as f:
    json.dump(comparison, f, indent=2)

#---Lưu artifact---

import joblib
import os

joblib.dump(model_clean, "artifacts/model_logreg_clean.pkl")
joblib.dump(vectorizer_clean, "artifacts/vectorizer_clean.pkl")
print("Save artifacts successully")