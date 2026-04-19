from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


@patch("app.main.init_db", new_callable=AsyncMock)
def test_health(_mock_init):
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
