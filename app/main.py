import os
import time
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]

model = None  # sẽ được gán lúc app khởi động
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Loading model from {model_uri} ...")
    start = time.time()
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded in {time.time() - start:.2f}s")
    yield
    # (chỗ này để dọn dẹp resource khi app tắt, nếu cần sau này)


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model not loaded",
        "model_loaded": model is not None,
    }