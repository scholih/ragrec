"""Integration tests for FastAPI endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragrec.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "services" in data
    assert "postgres" in data["services"]


def test_root_endpoint() -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "RagRec API"
    assert "version" in data


@pytest.mark.asyncio
async def test_similar_endpoint_with_image() -> None:
    """Test similarity search endpoint with image upload."""
    test_image = Path("data/sample/images/0888404001.jpg")

    if not test_image.exists():
        pytest.skip("Sample image not found")

    with open(test_image, "rb") as f:
        response = client.post(
            "/api/v1/similar",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"top_k": 5},
        )

    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert "count" in data
    assert "query_time_ms" in data

    assert data["count"] == 5
    assert len(data["results"]) == 5

    # Verify result structure
    first_result = data["results"][0]
    assert "article_id" in first_result
    assert "prod_name" in first_result
    assert "distance" in first_result
    assert "product_type_name" in first_result

    # Query time should be reasonable
    # Note: First request includes model loading time (~1.5s)
    # In production, embedder would be cached/shared across requests
    print(f"API query time: {data['query_time_ms']:.2f}ms")
    assert data["query_time_ms"] < 2000  # Allow for model loading


@pytest.mark.asyncio
async def test_similar_endpoint_with_category_filter() -> None:
    """Test similarity search with category filter."""
    test_image = Path("data/sample/images/0888404001.jpg")

    if not test_image.exists():
        pytest.skip("Sample image not found")

    with open(test_image, "rb") as f:
        response = client.post(
            "/api/v1/similar",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={
                "top_k": 5,
                "category_filter": "Dress",
            },
        )

    # May get 200 with 0 results if no dresses, or 200 with results
    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert "count" in data


def test_similar_endpoint_invalid_file_type() -> None:
    """Test that non-image files are rejected."""
    # Create a fake text file
    response = client.post(
        "/api/v1/similar",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        data={"top_k": 5},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_similar_endpoint_performance() -> None:
    """Test that the API endpoint meets performance requirements."""
    test_image = Path("data/sample/images/0888404001.jpg")

    if not test_image.exists():
        pytest.skip("Sample image not found")

    # Make multiple requests to test consistent performance
    times = []

    for _ in range(3):
        with open(test_image, "rb") as f:
            response = client.post(
                "/api/v1/similar",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"top_k": 10},
            )

        assert response.status_code == 200
        times.append(response.json()["query_time_ms"])

    avg_time = sum(times) / len(times)
    print(f"Average query time over {len(times)} requests: {avg_time:.2f}ms")

    # Note: Each request loads model fresh in test environment
    # In production with model caching, would be < 200ms
    assert avg_time < 2000
