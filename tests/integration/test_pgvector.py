"""Integration tests for pgvector store."""

import pytest

from ragrec.vectorstore import PgVectorStore


@pytest.mark.asyncio
async def test_pgvector_health_check() -> None:
    """Test pgvector health check."""
    async with PgVectorStore() as store:
        is_healthy = await store.health_check()
        assert is_healthy is True


@pytest.mark.asyncio
async def test_pgvector_search() -> None:
    """Test vector similarity search."""
    async with PgVectorStore() as store:
        # Get a real embedding from database to use as query
        # (in practice, this would come from encoding an image)
        import asyncpg

        async with store.pool.acquire() as conn:  # type: ignore
            row = await conn.fetchrow(
                "SELECT embedding FROM product_embeddings LIMIT 1"
            )
            # Extract vector values
            query_embedding = [float(x) for x in str(row["embedding"])[1:-1].split(",")]

        # Search for similar products
        results = await store.search(query_embedding, top_k=5)

        # Verify results
        assert len(results) == 5
        assert "article_id" in results.columns
        assert "distance" in results.columns
        assert "prod_name" in results.columns

        # First result should be the same embedding (distance ~0)
        assert results["distance"][0] < 0.01


@pytest.mark.asyncio
async def test_pgvector_search_with_filters() -> None:
    """Test vector search with metadata filters."""
    async with PgVectorStore() as store:
        # Get a sample embedding
        import asyncpg

        async with store.pool.acquire() as conn:  # type: ignore
            row = await conn.fetchrow(
                """
                SELECT pe.embedding, p.department_no
                FROM product_embeddings pe
                JOIN products p ON pe.article_id = p.article_id
                WHERE p.department_no IS NOT NULL
                LIMIT 1
                """
            )
            query_embedding = [float(x) for x in str(row["embedding"])[1:-1].split(",")]
            dept_no = row["department_no"]

        # Search with department filter
        results = await store.search(
            query_embedding, top_k=5, filters={"department_no": dept_no}
        )

        # All results should be from the same department
        assert len(results) > 0
        assert all(results["department_name"][0] == name for name in results["department_name"])


@pytest.mark.asyncio
async def test_pgvector_hybrid_search() -> None:
    """Test hybrid search (vector + text)."""
    async with PgVectorStore() as store:
        # Get a sample embedding
        import asyncpg

        async with store.pool.acquire() as conn:  # type: ignore
            row = await conn.fetchrow(
                "SELECT embedding FROM product_embeddings LIMIT 1"
            )
            query_embedding = [float(x) for x in str(row["embedding"])[1:-1].split(",")]

        # Hybrid search with text query
        results = await store.hybrid_search(
            query_embedding, query_text="shirt", top_k=5
        )

        # Verify results
        assert len(results) > 0
        assert "combined_score" in results.columns
        assert "text_boost" in results.columns


@pytest.mark.asyncio
async def test_index_stats() -> None:
    """Test getting index statistics."""
    async with PgVectorStore() as store:
        stats = await store.get_index_stats()

        assert stats["exists"] is True
        assert stats["total_embeddings"] > 0
        assert "size" in stats
