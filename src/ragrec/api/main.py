"""FastAPI application with health and recommendation endpoints."""

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ragrec.recommender import VisualRecommender
from ragrec.vectorstore import PgVectorStore

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


class ProductScore(BaseModel):
    """Product recommendation with similarity score."""

    article_id: int
    prod_name: str
    product_type_name: str
    colour_group_name: str
    department_name: str
    distance: float
    model_version: str


class SimilarityResponse(BaseModel):
    """Response for similarity search."""

    results: list[ProductScore]
    count: int
    query_time_ms: float


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns system status and service availability.
    """
    # Check PostgreSQL + pgvector
    postgres_healthy = False
    try:
        async with PgVectorStore() as store:
            postgres_healthy = await store.health_check()
    except Exception:
        pass

    return HealthResponse(
        status="ok" if postgres_healthy else "degraded",
        version="0.1.0",
        services={
            "postgres": postgres_healthy,
            "neo4j": False,  # Will check in Phase 2
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


@app.post("/api/v1/similar", response_model=SimilarityResponse)
async def find_similar_products(
    file: UploadFile = File(..., description="Product image file"),
    top_k: Annotated[int, Form()] = 10,
    category_filter: Annotated[str | None, Form()] = None,
) -> SimilarityResponse:
    """
    Find visually similar products by uploading an image.

    Args:
        file: Product image (JPEG, PNG)
        top_k: Number of similar products to return (default: 10)
        category_filter: Optional product type filter

    Returns:
        List of similar products with similarity scores
    """
    import time

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be JPEG or PNG.",
        )

    # Read image bytes
    image_bytes = await file.read()

    # Search for similar products
    start_time = time.perf_counter()

    async with VisualRecommender() as recommender:
        results_df = await recommender.find_similar(
            image_bytes=image_bytes,
            top_k=top_k,
            category_filter=category_filter,
        )

    query_time_ms = (time.perf_counter() - start_time) * 1000

    # Convert Polars DataFrame to response model
    products = [
        ProductScore(
            article_id=int(row["article_id"]),
            prod_name=str(row["prod_name"]),
            product_type_name=str(row["product_type_name"]),
            colour_group_name=str(row["colour_group_name"]),
            department_name=str(row["department_name"]),
            distance=float(row["distance"]),
            model_version=str(row["model_version"]),
        )
        for row in results_df.iter_rows(named=True)
    ]

    return SimilarityResponse(
        results=products,
        count=len(products),
        query_time_ms=query_time_ms,
    )
