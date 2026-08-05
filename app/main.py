import os
import time
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
import mlflow
from fastapi import FastAPI, BackgroundTasks
from dotenv import load_dotenv
from src.preprocessing import preprocess_pipeline, load_teencode_dict
from app.schemas import PredictRequest, PredictResponse
from app.db import insert_prediction_log

load_dotenv()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]

model = None  # sẽ được gán lúc app khởi động
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


teencode_map = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, teencode_map
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Loading model from {model_uri} ...")
    start = time.time()
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model loaded in {time.time() - start:.2f}s")
    teencode_map = load_teencode_dict("teencode_dict.json")
    yield
    # (chỗ này để dọn dẹp resource khi app tắt, nếu cần sau này)


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model not loaded",
        "model_loaded": model is not None,
    }

MODEL_VERSION_LABEL = f"{MODEL_NAME}@{MODEL_ALIAS}"  # đơn giản, dùng alias làm nhãn version

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    start = time.time()

    # Bước 1: preprocess — DÙNG LẠI y hệt hàm đã dùng lúc training, không viết logic mới
    cleaned_text = preprocess_pipeline(request.text, teencode_map)

    # Bước 2: predict — model là Pipeline (TF-IDF + Logistic Regression), nhận thẳng list text
    prediction = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    confidence = float(max(probabilities))  # xác suất của lớp được chọn

    latency_ms = (time.time() - start) * 1000

    # Bước 3: ghi log — chạy NGẦM sau khi response đã trả, không làm chậm client
    background_tasks.add_task(
        insert_prediction_log,
        input_text=request.text,
        sentiment=prediction,
        confidence=confidence,
        model_version=MODEL_VERSION_LABEL,
        latency_ms=latency_ms,
    )

    return PredictResponse(
        sentiment=prediction,
        confidence=confidence,
        model_version=MODEL_VERSION_LABEL,
        latency_ms=latency_ms,
    )


# pyrefly: ignore [missing-import]
from mlflow.exceptions import MlflowException
# pyrefly: ignore [missing-import]
from mlflow.tracking import MlflowClient
# pyrefly: ignore [missing-import]
from fastapi import HTTPException

from app.schemas import ModelInfoResponse

_mlflow_client = MlflowClient()  # tạo 1 lần ở module level, không tạo mới mỗi request


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    try:
        version_info = _mlflow_client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run = _mlflow_client.get_run(version_info.run_id)
    except MlflowException as e:
        # Alias không tồn tại, model chưa được đăng ký, hoặc MLflow server không phản hồi
        raise HTTPException(
            status_code=503,
            detail=f"Không lấy được metadata từ MLflow Registry: {e}",
        )

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=version_info.version,
        f1_score=run.data.metrics.get("f1"),
        accuracy=run.data.metrics.get("accuracy"),
        trained_at=version_info.creation_timestamp,
    )