# Sentiment Analysis API

An end-to-end sentiment analysis service for English movie reviews (IMDb), built to practice a full MLOps workflow: from raw data to a served, monitored API — not just a Jupyter notebook.

## Overview

Given a movie review in English, the API predicts whether the sentiment is **positive** or **negative**, along with a confidence score. The project focuses on doing the full lifecycle properly: reproducible preprocessing, experiment tracking, model versioning, and request logging for future monitoring.

**Scope:** English text only, binary classification (positive/negative — no neutral class, since the training data has no neutral ground truth).

## Architecture

```
[Training — run once]
IMDb dataset → custom preprocessing (teencode + emoji normalization)
             → TF-IDF vectorization → model training/selection
             → MLflow Model Registry (Production stage)

[Serving — every request]
Client → POST /predict
       → preprocess_pipeline() (identical function used at training time)
       → vectorize using the SAME fitted vectorizer from training
       → model.predict()
       → log request/response to PostgreSQL
       → return {sentiment, confidence, model_version}
```

Key design decisions:
- The exact same `preprocess_pipeline()` function is reused at both training and serving time, to avoid training-serving skew.
- The TF-IDF vectorizer is persisted and reloaded at serving time — never refit — so the feature space always matches the trained model.
- Every prediction is logged to PostgreSQL (input, output, model version, latency, timestamp), providing the raw data needed for future drift monitoring.

## Tech Stack
- **Modeling:** scikit-learn (TF-IDF, Logistic Regression / other classifiers)
- **Experiment tracking & model registry:** MLflow
- **Serving:** FastAPI
- **Storage/logging:** PostgreSQL
- **Packaging:** Docker, docker-compose
- **EDA:** pandas, matplotlib, wordcloud

## Project Structure
```
sentiment-api/
├── data/
│   ├── raw/                  # original IMDb CSV (not committed)
│   └── processed/            # preprocessed dataset
├── notebooks/
│   ├── eda_raw.ipynb          # EDA before preprocessing
│   └── eda_processed.ipynb    # EDA after preprocessing (comparison)
├── src/
│   ├── load_data.py           # data loading + validation
│   ├── preprocessing.py       # teencode + emoji normalization pipeline
│   ├── train.py               # training + MLflow logging
│   └── config.py
├── app/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py               # Pydantic request/response models
│   └── db.py                    # PostgreSQL connection
├── teencode_dict.json
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Preprocessing Pipeline
Custom preprocessing handles noise commonly found in real-world English text:
- **Teencode/abbreviations:** normalized via a hand-built JSON dictionary (`lol` → `laugh out loud`), matched with word-boundary regex to avoid partial matches.
- **Emoji:** translated to text using the `emoji` library, so sentiment-bearing emoji (😊😢) contribute to the model as tokens.
- **HTML noise:** IMDb reviews contain leftover `<br />` tags from the original crawl; these are stripped before training.

## API Endpoints

**`POST /predict`**
```json
// Request
{ "text": "This movie was absolutely wonderful, great acting!" }

// Response
{
  "sentiment": "positive",
  "confidence": 0.94,
  "model_version": "v2"
}
```

**`GET /health`** — service liveness check, confirms the model is loaded.

**`GET /model-info`** — returns current production model version, training metric, and training date.

## Getting Started

```bash
git clone <repo-url>
cd sentiment-api
docker compose up
```
This starts the FastAPI app, PostgreSQL, and the MLflow tracking server together.

## Results

> To be filled in after Week 2 (ablation study) and Week 4 (load testing) are complete.

| Metric | Value |
|---|---|
| Baseline (TF-IDF, no custom preprocessing) — F1 | 0.8951 |
| With custom preprocessing (teencode + emoji) — F1 | 0.8969 |
| API throughput (req/s, local load test) | TBD |
| API p95 latency | TBD |

> While yielding only a marginal improvement in model accuracy, the preprocessing pipeline is essential for standardizing messy API inputs, such as emojis and teencode.

## Limitations
- English only — not tested or intended for other languages.
- Binary classification only (positive/negative); no neutral class, since the training data provides no neutral ground truth to evaluate against.
- Trained on movie reviews (IMDb); may generalize poorly to very different domains (e.g. product reviews, social media slang beyond the covered teencode dictionary).

## Roadmap / Status
- [x] Week 1 — Data loading, custom preprocessing, EDA
- [ ] Week 2 — TF-IDF vectorization, model training, ablation study, MLflow registry
- [ ] Week 3 — FastAPI serving, PostgreSQL logging
- [ ] Week 4 — Dockerization, load testing, final documentation
