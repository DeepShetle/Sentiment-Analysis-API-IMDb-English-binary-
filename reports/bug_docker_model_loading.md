# Bug Report: Docker container không load được ML model

> **Ngày:** 2026-08-19  
> **Dự án:** Sentiment Analysis API  
> **Triệu chứng:** `GET /health` → `{"status": "model not loaded", "model_loaded": false}`

---

## 1. Triệu chứng

Khi chạy `docker compose up`, endpoint `/health` trả về `model_loaded: false`.  
Log container `app`:

```
Failed to load model from models:/sentiment-classifier@champion: No such artifact: ''
```

Model load thành công khi chạy **ngoài Docker** (trên Windows localhost), nhưng fail khi chạy **trong Docker**.

---

## 2. Quá trình điều tra

### Vòng 1 — Kiểm tra tổng quát

**Giả thuyết ban đầu:** Run artifact folder không tồn tại trên host.

Kiểm tra cấu trúc `./mlruns/1/` trên host:

```
mlruns/1/
  └── models/                                    ← ✅ CÓ
      ├── m-36d9f444430d4b8eb9f7baf74a93a7f4/    ← ✅ CÓ (model files 8.9MB)
      └── ... (7 models khác)
  └── 039b29d9d5df43b8a9e63bad2fae344c/          ← ❌ KHÔNG CÓ (run folder)
```

Kiểm tra DB (`mlflow.db`):

| Bảng | Trường | Giá trị |
|------|--------|---------|
| `registered_model_aliases` | champion → version | `2` |
| `model_versions` (v2) | source | `models:/m-36d9f444430d4b8eb9f7baf74a93a7f4` |
| `model_versions` (v2) | run_id | `039b29d9d5df43b8a9e63bad2fae344c` |
| `runs` (039b29d) | artifact_uri | `file:///mlflow/mlruns/1/039b29d.../artifacts` |
| `logged_models` (m-36d9f) | artifact_location | `file:///mlflow/mlruns/1/models/m-36d9f.../artifacts` |

**Kết luận vòng 1:** Run folder `039b29d.../` không có trên host → volume mount trống → MLflow không tìm thấy artifact.

---

### Vòng 2 — Tạo run artifact folder (thất bại)

Đã thực hiện:

```powershell
# Tạo thư mục run artifact
New-Item -ItemType Directory -Path ".\mlruns\1\039b29d9d5df43b8a9e63bad2fae344c\artifacts\model" -Force

# Copy model files vào
Copy-Item -Path ".\mlruns\1\models\m-36d9f444430d4b8eb9f7baf74a93a7f4\artifacts\*" `
    -Destination ".\mlruns\1\039b29d9d5df43b8a9e63bad2fae344c\artifacts\model\" -Recurse

docker compose restart app
```

**Kết quả:** Lỗi **vẫn y hệt** → `No such artifact: ''`

Xác nhận files đã tồn tại trong container:

```
docker compose exec app ls /mlflow/mlruns/1/039b29d.../artifacts/model/
→ conda.yaml  MLmodel  model.skops  python_env.yaml  registered_model_meta  requirements.txt
```

**Kết luận vòng 2:** Giả thuyết "thiếu run folder" là SAI. Vấn đề nằm ở chỗ khác.

---

### Vòng 3 — Trace chính xác luồng resolve (tìm ra root cause)

Chạy script debug bên trong container `app` để so sánh 2 cách gọi:

```python
# Cách 1: Load trực tiếp từ runs:/ → THÀNH CÔNG ✅
mlflow.sklearn.load_model('runs:/039b29d.../model')

# Cách 2: Load trực tiếp từ models:/m-<id> → THÀNH CÔNG ✅
mlflow.sklearn.load_model('models:/m-36d9f444430d4b8eb9f7baf74a93a7f4')

# Cách 3: Load qua alias → THẤT BẠI ❌
mlflow.sklearn.load_model('models:/sentiment-classifier@champion')
```

Monkey-patch `LocalArtifactRepository.__init__` để trace path:

```
models:/sentiment-classifier@champion
  → is_logged_model_uri = False
  → Dùng model_versions.storage_location
  → LocalArtifactRepository(artifact_uri='file:///D:/Coding/Independent_Projects/...')
  → _artifact_dir = /D:/Coding/Independent_Projects/...
  → exists = False ❌

models:/m-36d9f... (trực tiếp)
  → is_logged_model_uri = True
  → Dùng logged_models.artifact_location
  → LocalArtifactRepository(artifact_uri='file:///mlflow/mlruns/...')
  → _artifact_dir = /mlflow/mlruns/...
  → exists = True ✅
```

---

## 3. Nguyên nhân gốc

> **Trường `model_versions.storage_location` chứa Windows absolute path.**

```sql
SELECT storage_location FROM model_versions WHERE version=2;
-- Kết quả: file:///D:/Coding/Independent_Projects/Sentiment_Analysis_API/mlruns/1/models/m-36d9f.../artifacts
```

Khi Linux container parse URI `file:///D:/Coding/...`:
- Thành local path: `/D:/Coding/Independent_Projects/...`
- Path này **không tồn tại** trong container Linux
- → `No such artifact: ''`

### Tại sao `storage_location` lại chứa Windows path?

`register_model.py` được chạy **từ Windows localhost**:

```python
mlflow.set_tracking_uri("http://localhost:5000")          # ← gọi MLflow Docker qua port-forward
mlflow.register_model(model_uri="runs:/039b29d.../model") # ← register từ Windows
```

Khi `register_model()` thực thi, MLflow **client trên Windows** resolve artifact path theo filesystem Windows (`D:/Coding/...`), rồi gửi lên server để lưu vào `storage_location`. Server lưu nguyên path mà client gửi — dù server đang chạy trên Linux.

### Tại sao `logged_models.artifact_location` lại đúng?

Vì logged_model được tạo bởi `mlflow.sklearn.log_model()` trong `log_experiments.py` — lúc này MLflow **server** tự tạo artifact location dựa trên `--default-artifact-root /mlflow/mlruns` (container path).

### Tóm lại

| Hành động | Ai tạo path? | Path | Đúng/Sai |
|-----------|-------------|------|----------|
| `log_model()` → `logged_models.artifact_location` | **Server** (Linux) | `file:///mlflow/mlruns/...` | ✅ |
| `register_model()` → `model_versions.storage_location` | **Client** (Windows) | `file:///D:/Coding/...` | ❌ |

---

## 4. Cách sửa

### Fix đã áp dụng: UPDATE trực tiếp SQLite

File `fixbug.py`:

```python
import sqlite3
conn = sqlite3.connect('mlflow.db')
cur = conn.cursor()

cur.execute('''
    UPDATE model_versions 
    SET storage_location = 'file:///mlflow/mlruns/1/models/m-36d9f444430d4b8eb9f7baf74a93a7f4/artifacts'
    WHERE version = 2 AND name = 'sentiment-classifier'
''')
conn.commit()
conn.close()
```

Sau đó restart:

```bash
docker compose down
docker compose up
```

### Giải pháp dài hạn (phòng tránh tái phát)

| Giải pháp | Mô tả |
|-----------|--------|
| **Chạy `register_model.py` từ trong Docker** | Đảm bảo client resolve path theo Linux filesystem |
| **Bật `--serve-artifacts` trên MLflow server** | Server proxy artifact download qua HTTP, không phụ thuộc local path |
| **Dùng remote artifact store (S3/GCS)** | Path không phụ thuộc OS, hoạt động nhất quán mọi nơi |

---

## 5. Bài học rút ra

1. **MLflow lưu path theo OS của client**, không phải server. Khi client Windows register model vào server Linux → path bị mismatch.

2. **`model_versions` có 2 trường path khác nhau**: `source` (URI kiểu `models:/` hoặc `runs:/`) và `storage_location` (absolute file path). Khi resolve qua alias, MLflow dùng `storage_location`, KHÔNG dùng `logged_models.artifact_location`.

3. **Lỗi `No such artifact: ''`** không có nghĩa là "artifact path trống" — `''` là artifact **sub-path** mặc định (root). Lỗi thực sự là **thư mục root** (`_artifact_dir`) không tồn tại.

4. **Volume mount không tự tạo dữ liệu** — nếu Docker container ghi file vào path *trước khi* volume mount được gắn, dữ liệu chỉ tồn tại trong container ephemeral storage.
