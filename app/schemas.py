from pydantic import BaseModel, Field

#pydantic tự động từ chối khi input không hợp lệ, không cần tự viết if để check

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Review text in English")


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    model_version: str
    latency_ms: float

class ModelInfoResponse(BaseModel):
    model_name: str
    alias: str
    version: str
    f1_score: float | None
    accuracy: float | None
    trained_at: int  # unix timestamp (ms), lấy trực tiếp từ MLflow