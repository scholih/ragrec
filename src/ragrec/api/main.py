"""FastAPI application with health and recommendation endpoints."""

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ragrec.graph.client import Neo4jClient
from ragrec.recommender import CollaborativeRecommender, VisualRecommender
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

    # Check Neo4j
    neo4j_healthy = False
    try:
        async with Neo4jClient() as client:
            neo4j_healthy = await client.health_check()
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (postgres_healthy and neo4j_healthy) else "degraded",
        version="0.1.0",
        services={
            "postgres": postgres_healthy,
            "neo4j": neo4j_healthy,
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


# Graph-based recommendation endpoints


@app.get("/api/v1/graph/neighborhood/{product_id}")
async def get_product_neighborhood(
    product_id: int,
    depth: int = 2,
) -> dict:
    """
    Get product's graph neighborhood.

    Returns related products via visual similarity, category, and co-purchases.

    Args:
        product_id: Article ID of the product
        depth: Maximum traversal depth (default: 2, max: 3)

    Returns:
        Product neighborhood with similar, co-purchased, and same-category products
    """
    async with CollaborativeRecommender() as recommender:
        neighborhood = await recommender.get_product_neighborhood(
            product_id=product_id,
            depth=min(depth, 3),
        )

    if not neighborhood:
        raise HTTPException(
            status_code=404,
            detail=f"Product {product_id} not found in graph",
        )

    return neighborhood


@app.get("/api/v1/graph/journey/{customer_id}")
async def get_customer_journey(
    customer_id: str,
    limit: int = 50,
) -> dict:
    """
    Get customer's purchase journey over time.

    Args:
        customer_id: Customer ID
        limit: Maximum number of purchases to return (default: 50)

    Returns:
        Customer's purchase history ordered by timestamp
    """
    async with CollaborativeRecommender() as recommender:
        journey_df = await recommender.get_customer_journey(
            customer_id=customer_id,
            limit=limit,
        )

    return {
        "customer_id": customer_id,
        "purchases": journey_df.to_dicts(),
        "count": len(journey_df),
    }


@app.get("/api/v1/collaborative/{customer_id}")
async def get_collaborative_recommendations(
    customer_id: str,
    top_k: int = 10,
    min_shared_purchases: int = 2,
) -> dict:
    """
    Get collaborative filtering recommendations.

    Finds products purchased by customers with similar purchase history.

    Args:
        customer_id: Customer ID
        top_k: Number of recommendations (default: 10)
        min_shared_purchases: Minimum shared purchases to consider (default: 2)

    Returns:
        Recommended products with scores
    """
    async with CollaborativeRecommender() as recommender:
        recommendations_df = await recommender.recommend_for_customer(
            customer_id=customer_id,
            top_k=top_k,
            min_shared_purchases=min_shared_purchases,
        )

    return {
        "customer_id": customer_id,
        "recommendations": recommendations_df.to_dicts(),
        "count": len(recommendations_df),
    }


@app.get("/api/v1/complementary/{product_id}")
async def get_complementary_products(
    product_id: int,
    top_k: int = 5,
    min_co_purchases: int = 2,
) -> dict:
    """
    Get complementary product recommendations.

    Finds products frequently bought together with the given product.

    Args:
        product_id: Article ID of the product
        top_k: Number of recommendations (default: 5)
        min_co_purchases: Minimum co-purchases to consider (default: 2)

    Returns:
        Complementary products with co-purchase counts
    """
    async with CollaborativeRecommender() as recommender:
        complementary_df = await recommender.recommend_complementary(
            product_id=product_id,
            top_k=top_k,
            min_co_purchases=min_co_purchases,
        )

    return {
        "product_id": product_id,
        "complementary": complementary_df.to_dicts(),
        "count": len(complementary_df),
    }


@app.get("/api/v1/trending")
async def get_trending_products(
    days: int = 7,
    top_k: int = 10,
) -> dict:
    """
    Get trending products based on recent purchases.

    Args:
        days: Number of days to look back (default: 7)
        top_k: Number of products to return (default: 10)

    Returns:
        Trending products with recent purchase counts
    """
    async with CollaborativeRecommender() as recommender:
        trending_df = await recommender.get_trending(
            days=days,
            top_k=top_k,
        )

    return {
        "trending": trending_df.to_dicts(),
        "count": len(trending_df),
        "period_days": days,
    }


@app.get("/api/v1/graph/product/{product_id}/stats")
async def get_product_stats(product_id: int) -> dict:
    """
    Get statistics for a product.

    Args:
        product_id: Article ID of the product

    Returns:
        Product statistics including purchase count, unique customers, etc.
    """
    async with CollaborativeRecommender() as recommender:
        stats = await recommender.get_product_stats(product_id=product_id)

    if not stats:
        raise HTTPException(
            status_code=404,
            detail=f"Product {product_id} not found in graph",
        )

    return stats
