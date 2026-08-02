import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]  # crash sớm nếu thiếu biến env, không âm thầm dùng giá trị mặc định sai


def get_connection():
    """Tạo kết nối mới tới PostgreSQL. Dùng context manager khi gọi."""
    return psycopg2.connect(DATABASE_URL)


def insert_prediction_log(input_text: str, sentiment: str, confidence: float,
                            model_version: str, latency_ms: float) -> None:
    """Ghi 1 dòng log vào bảng prediction_logs."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_logs
                    (input_text, sentiment, confidence, model_version, latency_ms)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (input_text, sentiment, confidence, model_version, latency_ms),
            )
        conn.commit()
    finally:
        conn.close()