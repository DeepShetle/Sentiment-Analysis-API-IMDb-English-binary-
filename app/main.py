import os
import time
import logging
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, BackgroundTasks, HTTPException
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from src.preprocessing import preprocess_pipeline, load_teencode_dict
from app.schemas import PredictRequest, PredictResponse, ModelInfoResponse
from app.db import insert_prediction_log

logger = logging.getLogger(__name__)
load_dotenv()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]

model = None  # will be assigned during app startup
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


teencode_map = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, teencode_map
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Loading model from {model_uri} ...")
    start = time.time()
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}: {e}")
        model = None
    teencode_map = load_teencode_dict("teencode_dict.json")
    yield
    # (place to cleanup resources when app shuts down, if needed in the future)


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model not loaded",
        "model_loaded": model is not None,
    }

MODEL_VERSION_LABEL = f"{MODEL_NAME}@{MODEL_ALIAS}"  # simple, use alias as version label

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code = 503, detail = "Model is not loaded. The service is not ready to serve predictions.")
    
    start = time.time()
    try:
        # Step 1: preprocess — REUSE the exact function used during training, no new logic
        cleaned_text = preprocess_pipeline(request.text, teencode_map)

        # Step 2: predict — model is a Pipeline (TF-IDF + Logistic Regression), accepts a list of text directly
        prediction = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(max(probabilities))  # probability of the chosen class
    except Exception as e:
        logger.error(f"Prediction failed for input: {e}")
        raise HTTPException(status_code = 500, detail = "Failed to process the request.")

    latency_ms = (time.time() - start) * 1000

    # Step 3: logging — runs in BACKGROUND after response is returned, does not slow down client
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
_mlflow_client = MlflowClient()  # instantiate once at module level, not per request


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    try:
        version_info = _mlflow_client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run = _mlflow_client.get_run(version_info.run_id)
    except MlflowException as e:
        # Alias doesn't exist, model not registered, or MLflow server not responding
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch metadata from MLflow Registry: {e}",
        )

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=version_info.version,
        f1_score=run.data.metrics.get("f1"),
        accuracy=run.data.metrics.get("accuracy"),
        trained_at=version_info.creation_timestamp,
    )