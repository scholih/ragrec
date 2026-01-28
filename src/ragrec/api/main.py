"""FastAPI application with health and recommendation endpoints."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="RagRec API",
    description="Multi-modal e-retail recommendation system",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    services: dict[str, bool]


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns system status and service availability.
    """
    # TODO: Add actual service health checks in Phase 1
    return HealthResponse(
        status="ok",
        version="0.1.0",
        services={
            "postgres": False,  # Will check in Phase 1
            "neo4j": False,  # Will check in Phase 1
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "RagRec API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
