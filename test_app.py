import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from app import create_app


def test_liveness_has_service_metadata():
    response = TestClient(create_app()).get("/health/live")
    assert response.status_code == 200
    assert response.json()["service"] == "supermew"


def test_response_contains_request_diagnostics():
    response = TestClient(create_app()).get("/health/live", headers={"X-Request-ID": "health-check"})
    assert response.headers["X-Request-ID"] == "health-check"
    assert "X-Process-Time-ms" in response.headers
