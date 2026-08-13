#file text để tạo môi trường cho ứng dụng

FROM python:3.12-slim
# Xác định base image để xây app

WORKDIR /app    
# Thư mục làm việc mặc định là /app

# Cài dependencies trước, tách riêng khỏi COPY code —
# tận dụng Docker layer cache: nếu chỉ sửa code, không sửa requirements.txt,
# bước pip install không phải chạy lại (build nhanh hơn nhiều)
COPY requirements.txt .
# Copy file chứa thư viện cần dùng vào thư mục hiện tại "." (Ở đây là /app)
RUN pip install --no-cache-dir -r requirements.txt
# Tải về, không lưu cache để tối ưu dung lượng  

# Copy code cần thiết cho runtime — KHÔNG copy notebooks/, data/raw/ (không cần lúc serving)
COPY app/ ./app/
COPY src/ ./src/
COPY teencode_dict.json .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]