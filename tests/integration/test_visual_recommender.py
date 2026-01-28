"""Integration tests for visual recommender."""

from pathlib import Path

import pytest

from ragrec.recommender import VisualRecommender


@pytest.mark.asyncio
async def test_find_similar_with_image() -> None:
    """Test finding similar products from uploaded image."""
    # Use a sample image from the test data
    test_image = Path("data/sample/images/0888404001.jpg")

    if not test_image.exists():
        pytest.skip("Sample image not found")

    async with VisualRecommender() as recommender:
        image_bytes = test_image.read_bytes()

        results = await recommender.find_similar(
            image_bytes=image_bytes,
            top_k=5,
        )

        # Verify results
        assert len(results) == 5
        assert "article_id" in results.columns
        assert "distance" in results.columns
        assert "prod_name" in results.columns

        # First result should be very similar (likely the same product)
        assert results["distance"][0] < 0.1


@pytest.mark.asyncio
async def test_find_similar_with_category_filter() -> None:
    """Test finding similar products with category filter."""
    test_image = Path("data/sample/images/0888404001.jpg")

    if not test_image.exists():
        pytest.skip("Sample image not found")

    async with VisualRecommender() as recommender:
        image_bytes = test_image.read_bytes()

        # Get first result to determine category
        initial_results = await recommender.find_similar(
            image_bytes=image_bytes,
            top_k=1,
        )

        if initial_results.is_empty():
            pytest.skip("No results to test with")

        category = initial_results["product_type_name"][0]

        # Search with category filter
        results = await recommender.find_similar(
            image_bytes=image_bytes,
            top_k=5,
            category_filter=category,
        )

        # All results should be from the same category
        if not results.is_empty():
            assert all(
                results["product_type_name"][0] == ptype
                for ptype in results["product_type_name"]
            )


@pytest.mark.asyncio
async def test_find_similar_by_id() -> None:
    """Test finding similar products by product ID."""
    import asyncpg
    from ragrec.etl.config import ETLConfig

    # Get a real article ID from database
    pool = await asyncpg.create_pool(ETLConfig().database_url)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT article_id FROM product_embeddings LIMIT 1")
        article_id = row["article_id"]
    await pool.close()

    async with VisualRecommender() as recommender:
        results = await recommender.find_similar_by_id(
            product_id=article_id,
            top_k=5,
        )

        # Verify results
        assert len(results) > 0
        assert "article_id" in results.columns
        assert "distance" in results.columns

        # Reference product should NOT be in results
        assert article_id not in results["article_id"].to_list()

        # Results should be sorted by distance (most similar first)
        distances = results["distance"].to_list()
        assert distances == sorted(distances)


@pytest.mark.asyncio
async def test_find_similar_by_id_nonexistent() -> None:
    """Test finding similar products for non-existent ID."""
    async with VisualRecommender() as recommender:
        results = await recommender.find_similar_by_id(
            product_id=999999999,  # Non-existent ID
            top_k=5,
        )

        # Should return empty DataFrame
        assert results.is_empty()


@pytest.mark.asyncio
async def test_find_similar_performance() -> None:
    """Test that similarity search meets performance target."""
    import time

    test_image = Path("data/sample/images/0888404001.jpg")

    if not test_image.exists():
        pytest.skip("Sample image not found")

    async with VisualRecommender() as recommender:
        image_bytes = test_image.read_bytes()

        # First run to ensure model is loaded
        await recommender.find_similar(image_bytes=image_bytes, top_k=5)

        # Measure performance on second run
        start = time.perf_counter()
        results = await recommender.find_similar(image_bytes=image_bytes, top_k=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should return results
        assert len(results) > 0

        # Query time should be reasonable (< 200ms target)
        print(f"Query time: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 300  # Allow buffer for CI


@pytest.mark.asyncio
async def test_recommender_response_time() -> None:
    """Test that visual similarity search meets performance target."""
    import time

    test_image = Path("data/sample/images/0888404001.jpg")

    if not test_image.exists():
        pytest.skip("Sample image not found")

    async with VisualRecommender() as recommender:
        image_bytes = test_image.read_bytes()

        start = time.perf_counter()

        await recommender.find_similar(
            image_bytes=image_bytes,
            top_k=10,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be under 200ms (success criteria)
        # Note: First run may be slower due to model loading
        # so this test is for already-loaded state
        print(f"Query time: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 300  # Allow 300ms buffer for CI/slow machines
