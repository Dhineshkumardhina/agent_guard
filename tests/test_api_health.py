"""Integration test for FastAPI application lifecycle and health endpoint."""

from fastapi.testclient import TestClient
from backend.app.main import app


def test_api_health_endpoint():
    """Verify FastAPI server initializes and returns healthy status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["service"] == "agentguard"
        assert payload["project"] == "AgentGuard"


def test_api_root_endpoint():
    """Verify root endpoint exposes documentation and project metadata."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        payload = response.json()
        assert "AgentGuard" in payload["project"]
        assert payload["docs_url"] == "/docs"
