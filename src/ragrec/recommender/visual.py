"""Visual similarity-based recommender."""

from typing import Any

import polars as pl

from ragrec.embeddings import SigLIPEmbedder
from ragrec.vectorstore import PgVectorStore


class VisualRecommender:
    """Visual similarity recommender using SigLIP + pgvector."""

    def __init__(
        self,
        embedder: SigLIPEmbedder | None = None,
        vector_store: PgVectorStore | None = None,
    ) -> None:
        """Initialize visual recommender.

        Args:
            embedder: SigLIP embedder instance (creates default if None)
            vector_store: Vector store instance (creates default if None)
        """
        self.embedder = embedder or SigLIPEmbedder()
        self.vector_store = vector_store or PgVectorStore()
        self._owns_store = vector_store is None

    async def __aenter__(self) -> "VisualRecommender":
        """Async context manager entry."""
        if self._owns_store:
            await self.vector_store.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._owns_store:
            await self.vector_store.__aexit__(*args)

    async def find_similar(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> pl.DataFrame:
        """Find visually similar products from uploaded image.

        Args:
            image_bytes: Image file bytes
            top_k: Number of results to return
            category_filter: Optional product_type_name filter

        Returns:
            Polars DataFrame with columns: article_id, prod_name, distance, etc.
        """
        # Encode image to embedding
        embedding = self.embedder.encode_image(image_bytes)

        # Build filters
        filters: dict[str, Any] = {}
        if category_filter:
            filters["product_type_name"] = category_filter

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=embedding.tolist(),
            top_k=top_k,
            filters=filters if filters else None,
        )

        return results

    async def find_similar_by_id(
        self,
        product_id: int,
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> pl.DataFrame:
        """Find visually similar products to an existing product.

        Args:
            product_id: Article ID of the reference product
            top_k: Number of results to return (excluding the reference itself)
            category_filter: Optional product_type_name filter

        Returns:
            Polars DataFrame with similar products (excluding the reference)
        """
        # Get embedding for reference product
        query = """
            SELECT embedding
            FROM product_embeddings
            WHERE article_id = $1
        """

        async with self.vector_store.pool.acquire() as conn:  # type: ignore
            row = await conn.fetchrow(query, product_id)

            if not row:
                # Return empty DataFrame if product not found
                return pl.DataFrame()

            # Parse embedding from pgvector format "[1.0,2.0,...]"
            embedding_str = str(row["embedding"])
            embedding = [float(x) for x in embedding_str[1:-1].split(",")]

        # Build filters
        filters: dict[str, Any] = {}
        if category_filter:
            filters["product_type_name"] = category_filter

        # Search for similar products (will include the reference product itself)
        results = await self.vector_store.search(
            query_embedding=embedding,
            top_k=top_k + 1,  # Get one extra to account for reference
            filters=filters if filters else None,
        )

        # Remove the reference product from results
        results = results.filter(pl.col("article_id") != product_id)

        # Limit to top_k after filtering
        results = results.head(top_k)

        return results
