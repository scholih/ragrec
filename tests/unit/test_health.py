"""Tests for health endpoint."""

import pytest
from fastapi.testclient import TestClient

from ragrec.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Test health check endpoint returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "services" in data


def test_root_endpoint(client: TestClient) -> None:
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "RagRec API"
    assert data["version"] == "0.1.0"
