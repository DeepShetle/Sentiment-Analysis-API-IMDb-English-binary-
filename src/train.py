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