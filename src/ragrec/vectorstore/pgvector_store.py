"""pgvector-based vector store implementation."""

from typing import Any

import asyncpg
import polars as pl

from ragrec.etl.config import ETLConfig
from ragrec.vectorstore.base import VectorStore


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector vector store implementation."""

    def __init__(self, config: ETLConfig | None = None) -> None:
        """Initialize pgvector store.

        Args:
            config: ETL configuration with database URL
        """
        self.config = config or ETLConfig()
        self.pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "PgVectorStore":
        """Async context manager entry."""
        self.pool = await asyncpg.create_pool(
            self.config.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self.pool:
            await self.pool.close()

    async def upsert(
        self,
        ids: list[int],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update embeddings in database."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        count = 0
        async with self.pool.acquire() as conn:
            for i, (article_id, embedding) in enumerate(zip(ids, embeddings)):
                # Convert embedding to pgvector format
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                await conn.execute(
                    """
                    INSERT INTO product_embeddings (article_id, embedding, model_version)
                    VALUES ($1, $2::vector, $3)
                    ON CONFLICT (article_id) DO UPDATE
                    SET embedding = $2::vector, model_version = $3, created_at = CURRENT_TIMESTAMP
                    """,
                    article_id,
                    embedding_str,
                    metadata[i].get("model_version", "unknown") if metadata else "unknown",
                )
                count += 1

        return count

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Search for similar product embeddings using cosine distance.

        Args:
            query_embedding: Query vector (768-dim for SigLIP)
            top_k: Number of results to return
            filters: Optional filters (e.g., {"department_no": 1234})

        Returns:
            Polars DataFrame with columns: article_id, distance, prod_name, etc.
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        # Convert query embedding to pgvector format
        query_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build WHERE clause from filters
        where_clauses = []
        params = [query_str, top_k]
        param_num = 3

        if filters:
            for key, value in filters.items():
                where_clauses.append(f"p.{key} = ${param_num}")
                params.append(value)
                param_num += 1

        where_sql = ""
        if where_clauses:
            where_sql = "AND " + " AND ".join(where_clauses)

        # Query with cosine distance
        query_sql = f"""
            SELECT
                p.article_id,
                p.prod_name,
                p.product_type_name,
                p.colour_group_name,
                p.department_name,
                pe.embedding <=> $1::vector as distance,
                pe.model_version
            FROM product_embeddings pe
            JOIN products p ON pe.article_id = p.article_id
            WHERE 1=1 {where_sql}
            ORDER BY pe.embedding <=> $1::vector
            LIMIT $2
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)

        # Convert to Polars DataFrame
        if not rows:
            return pl.DataFrame()

        data = {
            "article_id": [row["article_id"] for row in rows],
            "prod_name": [row["prod_name"] for row in rows],
            "product_type_name": [row["product_type_name"] for row in rows],
            "colour_group_name": [row["colour_group_name"] for row in rows],
            "department_name": [row["department_name"] for row in rows],
            "distance": [float(row["distance"]) for row in rows],
            "model_version": [row["model_version"] for row in rows],
        }

        return pl.DataFrame(data)

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> pl.DataFrame:
        """Hybrid search combining vector similarity and text matching.

        Args:
            query_embedding: Query vector
            query_text: Text to match in product names
            top_k: Number of results to return

        Returns:
            Polars DataFrame with results ranked by combined score
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        # Convert query embedding to pgvector format
        query_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Hybrid search: vector similarity + text matching
        # Text match boosts score if product name contains query text
        query_sql = """
            SELECT
                p.article_id,
                p.prod_name,
                p.product_type_name,
                p.colour_group_name,
                p.department_name,
                pe.embedding <=> $1::vector as vector_distance,
                CASE
                    WHEN LOWER(p.prod_name) LIKE LOWER($3) THEN 0.7
                    WHEN LOWER(p.product_type_name) LIKE LOWER($3) THEN 0.8
                    ELSE 1.0
                END as text_boost,
                (pe.embedding <=> $1::vector) * CASE
                    WHEN LOWER(p.prod_name) LIKE LOWER($3) THEN 0.7
                    WHEN LOWER(p.product_type_name) LIKE LOWER($3) THEN 0.8
                    ELSE 1.0
                END as combined_score,
                pe.model_version
            FROM product_embeddings pe
            JOIN products p ON pe.article_id = p.article_id
            ORDER BY combined_score
            LIMIT $2
        """

        text_pattern = f"%{query_text}%"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query_sql, query_str, top_k, text_pattern)

        # Convert to Polars DataFrame
        if not rows:
            return pl.DataFrame()

        data = {
            "article_id": [row["article_id"] for row in rows],
            "prod_name": [row["prod_name"] for row in rows],
            "product_type_name": [row["product_type_name"] for row in rows],
            "colour_group_name": [row["colour_group_name"] for row in rows],
            "department_name": [row["department_name"] for row in rows],
            "vector_distance": [float(row["vector_distance"]) for row in rows],
            "text_boost": [float(row["text_boost"]) for row in rows],
            "combined_score": [float(row["combined_score"]) for row in rows],
            "model_version": [row["model_version"] for row in rows],
        }

        return pl.DataFrame(data)

    async def health_check(self) -> bool:
        """Check if PostgreSQL and pgvector are healthy."""
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                # Check pgvector extension is available
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
                )
                return result == 1
        except Exception:
            return False

    async def create_hnsw_index(
        self,
        ef_construction: int = 128,
        m: int = 16,
    ) -> None:
        """Create HNSW index for faster similarity search.

        Args:
            ef_construction: Size of dynamic candidate list (higher = better recall, slower build)
            m: Number of connections per layer (higher = better recall, more memory)
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            # Drop existing index if it exists
            await conn.execute(
                "DROP INDEX IF EXISTS idx_product_embeddings_hnsw"
            )

            # Create HNSW index with custom parameters
            await conn.execute(
                f"""
                CREATE INDEX idx_product_embeddings_hnsw
                ON product_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction})
                """
            )

    async def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the HNSW index."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            # Check if index exists
            idx_exists = await conn.fetchval(
                """
                SELECT COUNT(*) FROM pg_indexes
                WHERE indexname = 'idx_product_embeddings_hnsw'
                """
            )

            if not idx_exists:
                return {"exists": False}

            # Get index size
            idx_size = await conn.fetchval(
                """
                SELECT pg_size_pretty(pg_relation_size('idx_product_embeddings_hnsw'))
                """
            )

            # Get total embeddings count
            total_count = await conn.fetchval(
                "SELECT COUNT(*) FROM product_embeddings"
            )

            return {
                "exists": True,
                "size": idx_size,
                "total_embeddings": total_count,
            }
