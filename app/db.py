import os
import psycopg2
from dotenv import load_dotenv
import logging

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]  # crash early if env var is missing, do not fail silently


def get_connection():
    """Create a new PostgreSQL connection. Use as a context manager."""
    return psycopg2.connect(DATABASE_URL)

logger = logging.getLogger(__name__)

def insert_prediction_log(input_text: str, sentiment: str, confidence: float,
                            model_version: str, latency_ms: float) -> None:
    """Insert a log row into prediction_logs. Errors are suppressed here."""
    try:    # Suppress errors if PostgreSQL goes down
        conn = get_connection()
        try:    # Suppress insertion errors
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
        finally:    # Ensure connection is closed even on error
            conn.close()
    except Exception as e:
        logger.error(f"Failed to log prediction to database: {e}")