# Bài Học Xương Máu: Đưa MLflow vào Docker (Model Registry)

Quá trình đưa hệ thống MLflow (Tracking & Registry) vào Docker Compose thường gặp vô vàn cạm bẫy liên quan đến đường dẫn và hệ thống file. Dưới đây là 3 bài học đắt giá rút ra sau khi debug lỗi `No such artifact` kinh điển.

> [!WARNING]
> Nếu bạn dự định deploy MLflow Model Registry bằng Docker, hãy đọc kỹ tài liệu này trước khi bắt đầu để tiết kiệm hàng giờ đồng hồ debug!

---

## 🩸 Bài học 1: Bất đồng ngôn ngữ đường dẫn (Windows vs Linux)

**Triệu chứng:**
App API (chạy trong Docker/Linux) văng lỗi `No such artifact: 'D:/Coding/...'` khi gọi `mlflow.sklearn.load_model(models:/classifier@champion)`.

**Nguyên nhân:**
Khi bạn chạy script train và register model (`register_model.py`) trực tiếp từ **máy tính cá nhân (Windows)**, MLflow API sẽ lấy chính xác **đường dẫn tuyệt đối của Windows** (vd: `D:/.../mlruns`) và ghi chết vào database SQLite (`mlflow.db`) ở cột `storage_location`. 
Khi API app (chạy bằng Linux trong Docker) đọc database này, nó cố tìm đường dẫn ổ `D:/` bên trong container Linux và dĩ nhiên là thất bại.

**Cách khắc phục:**
**Tuyệt đối không register model từ host OS (Windows/Mac) nếu MLflow Server chạy trong Docker**. 
Hãy viết một script `bootstrap.py` (hoặc job) chạy **bên trong** một Docker container (như Alpine/Ubuntu) để thực hiện việc log và register. Khi đó, đường dẫn lưu vào DB sẽ luôn là chuẩn Linux (vd: `/mlflow/mlruns/...`).

---

## 🩸 Bài học 2: Ảo tưởng về "Đường dẫn chung" (Docker Volumes)

**Triệu chứng:**
App văng lỗi `No such artifact: ''` mặc dù đường dẫn trong DB đã chuẩn `/mlflow/mlruns/...`. Thậm chí log của script register báo đã tạo model thành công.

**Nguyên nhân:**
Script register (`bootstrap`) và app API là 2 container **độc lập**. 
Khi MLflow Server bảo `bootstrap`: *"Lưu model vào `/mlflow/mlruns/` đi"*, `bootstrap` sẽ lưu file vào ổ cứng cục bộ (ephemeral layer) của chính nó. Khi `app` khởi động, nó tìm trong `/mlflow/mlruns/` của nó thì trống rỗng. Mặc dù đường dẫn giống nhau, nhưng chúng là 2 thế giới song song.

**Cách khắc phục:**
Tất cả các container tham gia vào quá trình MLflow (MLflow Server, Job Register, App API) **BẮT BUỘC** phải được mount chung một Shared Volume.

```yaml
# Trong docker-compose.yml
volumes:
  - mlflow_data:/mlflow   # <-- Bắt buộc phải có dòng này ở tất cả container
```

---

## 🩸 Bài học 3: Cạm bẫy bảo mật "Chỉ Đọc" (Read-Only Mounts)

**Triệu chứng:**
App tìm thấy model nhưng văng lỗi: `Failed to load model... [Errno 30] Read-only file system: '.../artifacts/registered_model_meta'`

**Nguyên nhân:**
Theo tư duy bảo mật thông thường, container API chỉ cần *đọc* model, nên ta set volume mount là read-only (`mlflow_data:/mlflow:ro`). 
Tuy nhiên, hàm `mlflow.sklearn.load_model()` của MLflow khi tải model từ local filesystem sẽ **cố gắng ghi một vài file tạm/cache** (vd: `registered_model_meta`) vào chính thư mục chứa artifact đó để đánh dấu. Cờ `:ro` đã chặn hành động ghi này làm ứng dụng crash.

**Cách khắc phục:**
Không được sử dụng cờ `:ro` (read-only) khi mount thư mục artifact của MLflow cho các container cần load model. Phải cấp quyền ghi (`rw` - mặc định).

```diff
-  volumes:
-    - mlflow_data:/mlflow:ro    # SAI: Sẽ gây lỗi Errno 30
+  volumes:
+    - mlflow_data:/mlflow       # ĐÚNG: Phải cho phép MLflow ghi cache
```

---

### Tổng kết mô hình chuẩn mực:

1. Dùng **Named Volume** (`mlflow_data`).
2. **Mount volume này vào mọi container**: `mlflow`, `bootstrap` (job train/register), `app` (API).
3. Đảm bảo mount **có quyền ghi**.
4. Chạy mọi logic tương tác với MLflow (Log, Register, Load) **hoàn toàn bên trong Docker network**, dùng chung biến môi trường: `MLFLOW_TRACKING_URI=http://mlflow:5000`.
