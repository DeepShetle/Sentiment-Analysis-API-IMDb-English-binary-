from load_data import load_imdb_data
from sklearn.model_selection import train_test_split #Hàm chia data thành tập train và tập test

RANDOM_STATE = 42 #Dùng để đảm bảo rằng kết quả chia train test là như nhau mỗi lần chạy
TEST_SIZE = 0.2 #Dùng để xác định kích thước của tập test, ở đây là 20%

df = load_imdb_data("data/processed/imdb_processed.csv") 

df["review_baseline"] = df["review"].str.lower()    #Chỉ lowercase, chưa clean

X_train, X_test, y_train, y_test = train_test_split(
    df["review_baseline"],
    df["sentiment"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["sentiment"],       #chia đều theo cột sentiment (Tỉ lệ pos/neg ở train và test là như nhau(50/50))
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

# 4. Reset index để tránh lỗi lệch index ở các bước sau (đặc biệt khi ghép lại với DataFrame khác)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


#--- TF-IDF vectorize ---

from sklearn.feature_extraction.text import TfidfVectorizer

# Check vocab size
train_vocab = set(" ".join(X_train).split())    #Ghép thành 1 chuỗi duy nhất rồi split ra
print(f"Vocab size thực tế của tập train (word-level, chưa qua TF-IDF): {len(train_vocab)}")

MAX_FEATURES = 10000

vectorizer = TfidfVectorizer(
    max_features = MAX_FEATURES,
    lowercase = False
)

X_train_vec = vectorizer.fit_transform(X_train)     #fit: Học từ vựng và tính IDF (Độ hiếm của từ) dựa trên dữ liệu trong X_train
            #transform dùng vocab và IDF vừa học được để chuyển từng review trong X_train thành 1 vector số, mỗi cột tương ứng với 1 từ, giả trị là điểm TF-IDF
X_test_vec = vectorizer.transform(X_test)   #Không fit, tức là dùng lại bộ vocab + IDF đã học từ X_train, áp lên X_test để transform
#Nếu trong test có từ ko có trong từ điển, tự động bỏ qua -> tránh lỗi Runtime Error

#Check bằng mắt
print(f"Shape X_train_vec: {X_train_vec.shape}")
print(f"Shape X_test_vec: {X_test_vec.shape}")

assert X_train_vec.shape[1] == X_test_vec.shape[1], "Số features giữa train/test phải khớp"
    #Shape[1] là số cột
    #Shape[2] là số dòng
assert X_train_vec.shape[0] == len(y_train)
assert X_test_vec.shape[0] == len(y_test)

import numpy as np
feature_names = vectorizer.get_feature_names_out()  #Lấy ra danh sách tên các từ ứng với từng cột trong ma trận X_train_vec/X_test_vec

avg_tfidf = np.asarray(X_train_vec.mean(axis=0)).flatten()
                #X_train_vec.mean(axis=0): tính TB cộng theo cột -> thành ma trận 2 chiều (1 dòng, N cột)
                #np.asarray(): Chuyển về mảng numpy
                #flatten(): Chuyển về mảng 1 chiều

top_20_idx = avg_tfidf.argsort()[-20:][::-1]
#argsort(): Trả về vị trí mà sẽ xếp mảng theo tăng dần, không phải trả về giá trị
#[-20]: Lấy 20 giá trị cuối (Chỉ số của 20 giá trị lớn nhất)
#[::-1]: Đảo ngược sắp xếp min -> max thành max -> min

print("Top 20 từ theo TF-IDF trung bình (baseline, chưa preprocess):")
for i in top_20_idx:
    print(f"  {feature_names[i]}: {avg_tfidf[i]:.4f}")

#--- Train baseline (logistic regression) ---

from sklearn.linear_model import LogisticRegression
import warnings

# Bắt warning thành lỗi tạm thời để phát hiện ConvergenceWarning ngay,
# thay vì bỏ qua và không để ý model có hội tụ hay không
with warnings.catch_warnings():     #Bẫy lại warning để nó không chạy tiếp nếu model chưa hội tụ
    warnings.filterwarnings("error", category=Warning)      #Biến Warning thành lỗi
    try:
        model_baseline = LogisticRegression(
            max_iter=1000,      #Chọn số vòng lặp tối đa để tìm bộ trọng số tối ưu
            random_state=RANDOM_STATE,
        )
        model_baseline.fit(X_train_vec, y_train)    #vì warning đã được coi là lỗi -> nếu xảy ra thì nhảy vào except
        print("Model hội tụ thành công trong giới hạn max_iter=1000")
    except Warning as w:
        print(f"CẢNH BÁO: {w}")     #In ra warning
        print("Cân nhắc tăng max_iter (ví dụ 2000) — nếu tăng, phải dùng CÙNG giá trị này ở Ngày 7")     
        # Train lại bình thường (không raise) để vẫn có model dùng tiếp
        model_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model_baseline.fit(X_train_vec, y_train)

# Kiểm tra bắt buộc: model đã học được bao nhiêu "lớp" nhãn
print(f"Các lớp model học được: {model_baseline.classes_}")
assert list(model_baseline.classes_) == sorted(y_train.unique()), \
    "Model phải học đúng 2 lớp positive/negative"

#---Đánh giá - accuracy + F1 ---

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
#import các chỉ số đo lường

y_pred = model_baseline.predict(X_test_vec)     #Dự đoán nhãn cho tập X_test_vec

# Kiểm tra bắt buộc: số lượng dự đoán phải khớp số lượng mẫu test
assert len(y_pred) == len(y_test), "Số dự đoán phải bằng số mẫu test"

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label="positive")

print(f"Baseline — Accuracy: {acc:.4f}, F1: {f1:.4f}")
print("\nClassification report chi tiết:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=["negative", "positive"]))

#---Lưu kết quả có cấu trúc rõ ràng ---
import json
import os

os.makedirs("reports", exist_ok=True)

result_entry = {
    "run": "baseline_no_preprocessing",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "max_features": MAX_FEATURES,
    "accuracy": round(acc, 4),
    "f1": round(f1, 4),
}

results_path = "reports/ablation_results.json"

# Đọc file cũ nếu đã tồn tại, để append thay vì ghi đè
if os.path.exists(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = []

all_results.append(result_entry)

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Đã lưu kết quả vào {results_path}")


#---Lưu artifact (model + vectorizer) ---

import joblib   #Dùng để lưu model và vectorizer
import os

os.makedirs("artifacts", exist_ok=True)

joblib.dump(model_baseline, "artifacts/model_baseline.pkl")
joblib.dump(vectorizer, "artifacts/vectorizer_baseline.pkl")

print("Đã lưu model_baseline.pkl và vectorizer_baseline.pkl vào artifacts/")

# Kiểm tra bắt buộc: load lại thử để xác nhận file lưu đúng, không bị hỏng
loaded_model = joblib.load("artifacts/model_baseline.pkl")
loaded_vectorizer = joblib.load("artifacts/vectorizer_baseline.pkl")

# So sánh dự đoán giữa model gốc và model vừa load lại — phải giống hệt nhau
sample_check = loaded_model.predict(X_test_vec[:5])
original_check = model_baseline.predict(X_test_vec[:5])
assert list(sample_check) == list(original_check), \
    "Model sau khi load lại phải cho kết quả dự đoán giống hệt model gốc"

print("Xác nhận: load lại artifact thành công, dự đoán khớp với model gốc")
