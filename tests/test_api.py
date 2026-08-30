import pytest

def test_health_returns_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

@pytest.mark.parametrize(
    "text",
    [
        "This movie was absolutely fantastic!",
        "Waste of time, terrible acting.",
    ],
)
def test_predict_returns_valid_response(client, text):  # Test if response has valid format
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ("positive", "negative")
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_version" in data

def test_predict_rejects_empty_text(client): # Test sending empty text
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_predict_rejects_missing_field(client): # Test missing 'text' field
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_handles_teencode_and_emoji(client): # Test text with teencode and emoji
    response = client.post("/predict", json={"text": "lol this movie was great 😊 fr fr"})
    assert response.status_code == 200

def test_model_info_returns_metadata(client): # Test fetching metadata from MLflow Registry
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "sentiment-classifier"
    assert data["alias"] == "champion"
    assert data["f1_score"] is not None
    assert data["accuracy"] is not None
    assert data["version"] is not None

def test_predict_returns_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr("app.main.model", None)
    response = client.post("/predict", json={"text": "great movie"})
    assert response.status_code == 503

def test_predict_survives_logging_failure(client, monkeypatch):
    def broken_get_connection():
        raise ConnectionError("Simulated DB failure")
    monkeypatch.setattr("app.db.get_connection", broken_get_connection)

    response = client.post("/predict", json={"text": "great movie"})
    assert response.status_code == 200