import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client(monkeypatch):
    # Prevent actual logging to Postgres during tests —
    # automated tests should not create garbage data in the real DB
    monkeypatch.setattr("app.main.insert_prediction_log", lambda **kwargs: None)
    with TestClient(app) as c:
        yield c