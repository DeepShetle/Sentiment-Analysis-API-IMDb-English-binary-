CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    input_text TEXT NOT NULL,
    sentiment VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    latency_ms FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);