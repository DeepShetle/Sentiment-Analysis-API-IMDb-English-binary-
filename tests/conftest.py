import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client(monkeypatch):
    # Chặn việc ghi log thật vào Postgres trong lúc test —
    # test tự động không nên tạo dữ liệu rác trong DB thật mỗi lần chạy
    monkeypatch.setattr("app.main.insert_prediction_log", lambda **kwargs: None)
    with TestClient(app) as c:
        yield c