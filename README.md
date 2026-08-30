# Sentiment Analysis API

An end-to-end sentiment analysis service for English movie reviews (IMDb), built to practice a full MLOps workflow: from raw data to a served, monitored API — not just a Jupyter notebook.

## Overview

Given a movie review in English, the API predicts whether the sentiment is **positive** or **negative**, along with a confidence score. The project focuses on doing the full lifecycle properly: reproducible preprocessing, experiment tracking, model versioning, and request logging for future monitoring.

**Scope:** English text only, binary classification (positive/negative — no neutral class, since the training data has no neutral ground truth).

## Architecture

```
[Training — Local environment]
IMDb dataset → custom preprocessing (teencode + emoji normalization)
             → TF-IDF vectorization → model training/selection
             → log models & metrics to local MLflow Tracking

[Deployment & Serving — Docker Compose]
1. mlflow (container)     : MLflow Server + SQLite + local artifact storage.
2. bootstrap (container)  : One-time job. Registers the best pre-trained model (from local /artifacts directory) to MLflow Model Registry as 'champion'.
3. postgres (container)   : Database for API request/response logging.
4. app (container)        : FastAPI application.

Client → POST /predict
       → [app] Fetches champion model & vectorizer from MLflow Registry (cached at startup)
       → [app] preprocess_pipeline() (identical function used at training time)
       → [app] vectorize using the SAME fitted vectorizer
       → [app] model.predict()
       → [app] log request/response asynchronously to PostgreSQL
       → return {sentiment, confidence, model_version}
```

Key design decisions:
- **Separation of Concerns:** MLflow handles model versioning and registry, PostgreSQL handles application-level logging, and FastAPI handles serving.
- **Training-Serving Skew Prevention:** The exact same `preprocess_pipeline()` function is reused at both training and serving time. The TF-IDF vectorizer is persisted and reloaded at serving time (never refit).
- **Asynchronous Logging:** Every prediction is logged to PostgreSQL (input, output, model version, latency, timestamp) in a background task, ensuring it never adds latency to the client response. Logging failures do not fail the prediction request.
- **Dynamic Model Loading:** The FastAPI app fetches the "champion" model directly from the MLflow registry at startup instead of hardcoding a specific file path. A bootstrap container seeds this registry on startup.

## Tech Stack
- **Modeling:** scikit-learn (TF-IDF, Logistic Regression, SVM, Random Forest)
- **Experiment tracking & model registry:** MLflow (local, SQLite backend)
- **Serving:** FastAPI
- **Storage/logging:** PostgreSQL (Dockerized, local development)
- **Packaging:** Docker, docker-compose
- **Preprocessing:** pandas, `emoji` library, regex
- **EDA:** matplotlib, wordcloud
- **Testing:** pytest, locust


## Project Structure
```
sentiment-api/
├── app/
│   ├── main.py                # FastAPI app & endpoints
│   ├── schemas.py             # Pydantic request/response models
│   └── db.py                  # PostgreSQL connection
├── artifacts/                 # Serialized models and vectorizers (.pkl)
├── data/
│   ├── raw/                   # Original IMDb CSV (not committed)
│   └── processed/             # Preprocessed dataset
├── locust/                    # Load testing scripts
│   ├── locustfile.py
│   └── sample_reviews.py
├── notebooks/
│   ├── eda_raw.ipynb          # EDA before preprocessing
│   └── eda_processed.ipynb    # EDA after preprocessing (comparison)
├── reports/                   # Performance metrics, loadtest results, and EDA charts
├── sql/
│   └── init.sql               # PostgreSQL initialization script
├── src/
│   ├── bootstrap.py           # Registers champion model to MLflow on Docker startup
│   ├── config.py
│   ├── load_data.py           # Data loading + validation
│   ├── log_experiments.py     # Log all 4 runs to MLflow
│   ├── preprocessing.py       # Teencode + emoji normalization pipeline
│   ├── register_model.py      # Register champion model to MLflow Registry (local script)
│   ├── train_logistic.py      # Train logistic with preprocessing
│   ├── train_nopreprocess.py  # Train logistic baseline
│   └── train_svm_rf.py        # Train SVM and Random Forest with preprocessing
├── tests/
│   ├── conftest.py
│   ├── test_api.py            # Integration tests for FastAPI endpoints
│   └── test_preprocessing.py  # Unit tests for preprocessing logic
├── teencode_dict.json         # Dictionary for normalizing slang/abbreviations
├── docker-compose.yml         # Defines MLflow, Postgres, App, and Bootstrap services
├── Dockerfile                 # Docker image for both FastAPI app and Bootstrap script
├── requirements.txt           # Python dependencies
└── README.md
```

## Preprocessing Pipeline
Custom preprocessing handles noise commonly found in real-world English text:
- **Teencode/abbreviations:** normalized via a hand-built JSON dictionary (`lol` → `laugh out loud`), matched with word-boundary regex to avoid partial matches.
- **Emoji:** translated to text using the `emoji` library, so sentiment-bearing emoji (😊😢) contribute to the model as tokens.
- **HTML noise:** IMDb reviews contain leftover `<br />` tags from the original crawl; these are stripped before training.

## API Endpoints

**`POST /predict`** — classify the sentiment of a review.
```json
// Request
{ "text": "This movie was absolutely wonderful, great acting!" }

// Response (200 OK)
{
  "sentiment": "positive",
  "confidence": 0.94,
  "model_version": "sentiment-classifier@champion",
  "latency_ms": 12.3
}
```
Input is passed through the exact same `preprocess_pipeline()` used at training time before being vectorized and scored by the production model. Every successful prediction is logged asynchronously to PostgreSQL (via a background task) so it never adds latency to the response.

**`GET /health`** — liveness check.
```json
{ "status": "ok", "model_loaded": true }
```
Confirms the service is up and the production model has been loaded into memory.

**`GET /model-info`** — metadata about the model currently serving traffic, pulled live from the MLflow Registry (not hardcoded).
```json
{
  "model_name": "sentiment-classifier",
  "alias": "champion",
  "version": "2",
  "f1_score": 0.8969,
  "accuracy": 0.8953,
  "trained_at": 1732000000000
}
```

**Error handling**

| Status | When |
|---|---|
| `422 Unprocessable Entity` | Request body fails validation (missing/empty `text`) |
| `503 Service Unavailable` | Model failed to load at startup (e.g. MLflow unreachable) — request rejected before touching a `None` model |
| `500 Internal Server Error` | Unexpected failure during preprocessing/inference — the client receives a generic message; details are logged server-side, not exposed in the response |

A logging failure (e.g. PostgreSQL temporarily down) never surfaces as an error to the client — `/predict` still returns the prediction normally, and the failure is only logged server-side. Logging is a monitoring concern, not a correctness dependency of the core feature.

## Dataset (For Training/Re-running Notebooks)

Because the raw dataset is large, it is ignored via `.gitignore` and not included in this repository. If you want to run the Jupyter notebooks or retrain the models yourself, you need to download the dataset manually:

1. Download the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.
2. Save the extracted `.csv` file as `IMDB Dataset.csv` and place it inside the `data/raw/` directory.

## Getting Started

Requires Docker Desktop.

```bash
git clone https://github.com/DeepShetle/Sentiment-Analysis-API-IMDb-English-binary-.git sentiment-api
cd sentiment-api
docker compose up
```

This starts PostgreSQL, MLflow (tracking + registry), a one-time bootstrap job
that registers the pre-trained model, and the FastAPI app — all in one command.
First run may take a few minutes while the bootstrap job registers the model
and MLflow installs its dependencies. Once ready, visit:

- `http://localhost:8000/docs` — interactive API docs (Swagger UI)
- `http://localhost:5000` — MLflow UI (experiment comparison, model registry)
```

## Results

| Metric | Value |
|---|---|
| Baseline (TF-IDF, no custom preprocessing) — F1 | 0.8951 |
| With custom preprocessing (teencode + emoji) — F1 | 0.8969 |
| API throughput (req/s, local load test) | 135 req/s |
| API p95 latency | 71 ms |

> While the accuracy improvement was marginal on this dataset, the preprocessing pipeline is expected to matter more for informal user input at inference time — this is a hypothesis to validate once real usage logs are available.
> Tested locally via Locust (50 concurrent users, 2-minute run) on a single machine
> Running the full Docker Compose stack (app, MLflow, PostgreSQL) alongside the load
> Generator — not representative of production infrastructure with dedicated resources.

## Model Training & Experiment Tracking

Three algorithms were trained and compared on the same preprocessed
train/test split (`random_state=42`, `max_features=10000`):

| Model               | Accuracy | F1     | Train time (s) |
|---------------------|----------|--------|-----------------|
| Logistic Regression | 0.8953   | 0.8969 | 0.55            |
| LinearSVC (uncalibrated) | 0.8923 | 0.8934 | 1.86            |
| Random Forest        | 0.8405   | 0.8404 | 17.98           |

All experiments were logged to MLflow Tracking, and the best-performing
model was registered in the MLflow Model Registry:

**Registered model in MLflow Registry: `sentiment-classifier`, alias
`champion` → Logistic Regression on `review_clean` (accuracy 0.8953,
F1 0.8969) — outperforms SVM (0.8923/0.8934) and Random Forest
(0.8405/0.8404) on both accuracy and F1.**

## Limitations
- English only — not tested or intended for other languages.
- Binary classification only (positive/negative); no neutral class, since the training data provides no neutral ground truth to evaluate against.
- Trained on movie reviews (IMDb); may generalize poorly to very different domains (e.g. product reviews, social media slang beyond the covered teencode dictionary).

## Roadmap / Status
- [x] Week 1 — Data loading, custom preprocessing, EDA
- [x] Week 2 — TF-IDF vectorization, model training, ablation study, MLflow registry
- [x] Week 3 — FastAPI serving, PostgreSQL logging
- [x] Week 4 — Dockerization, load testing, final documentation
