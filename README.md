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
             → alias champion

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
- **Modeling:** scikit-learn (TF-IDF, Logistic Regression, SVM, Random Forest)
- **Experiment tracking & model registry:** MLflow (local, SQLite backend)
- **Serving:** FastAPI
- **Storage/logging:** PostgreSQL (Dockerized, local development)
- **Packaging:** Docker, docker-compose
- **Preprocessing:** pandas, `emoji` library, regex
- **EDA:** matplotlib, wordcloud
- **Testing:** pytest


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
│   ├── train_nopreprocess.py               # train logistic baseline
│   ├── train_logistic.py                   # train logistic with preprocessing
│   ├── train_svm_rf.py                     # train svm and random forest with preprocessing
│   ├── log_experiments.py                  # Log all 4 runs to MLflow
│   ├── register_model.py                   # Register champion model to MLflow Registry
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
Coming in Week 3

## Getting Started
Coming in Week 4

## Results

> To be filled in after Week 4 (load testing) are complete.

| Metric | Value |
|---|---|
| Baseline (TF-IDF, no custom preprocessing) — F1 | 0.8951 |
| With custom preprocessing (teencode + emoji) — F1 | 0.8969 |
| API throughput (req/s, local load test) | TBD |
| API p95 latency | TBD |

> While the accuracy improvement was marginal on this dataset, the preprocessing pipeline is expected to matter more for informal user input at inference time — this is a hypothesis to validate once real usage logs are available.

## Model Training & Experiment Tracking

Three algorithms were trained and compared on the same preprocessed
train/test split (`random_state=42`, `max_features=10000`):

| Model               | Accuracy | F1     | Train time (s) |
|---------------------|----------|--------|-----------------|
| Logistic Regression | 0.8953   | 0.8969 | 0.55            |
| LinearSVC (calibrated) | 0.8923 | 0.8934 | 1.86            |
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
- [ ] Week 4 — Dockerization, load testing, final documentation
