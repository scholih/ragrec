"""Collaborative filtering recommender using Neo4j graph."""

from typing import Any

import polars as pl

from ragrec.graph.client import Neo4jClient
from ragrec.graph.queries import GraphQueries


class CollaborativeRecommender:
    """Graph-based collaborative filtering recommender."""

    def __init__(self, neo4j_client: Neo4jClient | None = None) -> None:
        """Initialize collaborative recommender.

        Args:
            neo4j_client: Neo4j client (creates default if None)
        """
        self.neo4j_client = neo4j_client or Neo4jClient()
        self._owns_client = neo4j_client is None
        self.queries = GraphQueries(self.neo4j_client)

    async def __aenter__(self) -> "CollaborativeRecommender":
        """Async context manager entry."""
        if self._owns_client:
            await self.neo4j_client.__aenter__()
        self.queries = GraphQueries(self.neo4j_client)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._owns_client:
            await self.neo4j_client.__aexit__(*args)

    async def recommend_for_customer(
        self,
        customer_id: str,
        top_k: int = 10,
        min_shared_purchases: int = 2,
    ) -> pl.DataFrame:
        """Get collaborative recommendations for a customer.

        Finds products purchased by customers with similar purchase history.

        Args:
            customer_id: Customer ID
            top_k: Number of recommendations to return
            min_shared_purchases: Minimum shared purchases to consider customers similar

        Returns:
            Polars DataFrame with columns: product_id, name, category, score
        """
        results = await self.queries.get_collaborative_recommendations(
            customer_id=customer_id,
            top_k=top_k,
            min_shared_purchases=min_shared_purchases,
        )

        if not results:
            return pl.DataFrame(
                schema={
                    "product_id": pl.Int64,
                    "name": pl.Utf8,
                    "category": pl.Utf8,
                    "score": pl.Float64,
                }
            )

        # Convert Neo4j results to Polars DataFrame
        rows = []
        for result in results:
            product = result["product"]
            category = result.get("category", {})
            score = result["score"]

            rows.append({
                "product_id": product["id"],
                "name": product["name"],
                "category": category.get("name") if category else None,
                "score": float(score),
            })

        return pl.DataFrame(rows)

    async def recommend_complementary(
        self,
        product_id: int,
        top_k: int = 5,
        min_co_purchases: int = 2,
    ) -> pl.DataFrame:
        """Get complementary product recommendations.

        Finds products frequently bought together with the given product.

        Args:
            product_id: Article ID of the product
            top_k: Number of recommendations to return
            min_co_purchases: Minimum co-purchases to consider

        Returns:
            Polars DataFrame with columns: product_id, name, category, co_purchase_count, score
        """
        results = await self.queries.find_complementary_products(
            product_id=product_id,
            top_k=top_k,
            min_co_purchases=min_co_purchases,
        )

        if not results:
            return pl.DataFrame(
                schema={
                    "product_id": pl.Int64,
                    "name": pl.Utf8,
                    "category": pl.Utf8,
                    "co_purchase_count": pl.Int64,
                    "score": pl.Float64,
                }
            )

        # Convert Neo4j results to Polars DataFrame
        rows = []
        for result in results:
            product = result["product"]
            category = result.get("category", {})

            rows.append({
                "product_id": product["id"],
                "name": product["name"],
                "category": category.get("name") if category else None,
                "co_purchase_count": result["co_purchase_count"],
                "score": result["score"],
            })

        return pl.DataFrame(rows)

    async def get_trending(
        self,
        days: int = 7,
        top_k: int = 10,
    ) -> pl.DataFrame:
        """Get trending products based on recent purchases.

        Args:
            days: Number of days to look back
            top_k: Number of products to return

        Returns:
            Polars DataFrame with columns: product_id, name, category, recent_purchases, avg_price
        """
        results = await self.queries.get_trending_products(
            days=days,
            top_k=top_k,
        )

        if not results:
            return pl.DataFrame(
                schema={
                    "product_id": pl.Int64,
                    "name": pl.Utf8,
                    "category": pl.Utf8,
                    "recent_purchases": pl.Int64,
                    "avg_price": pl.Float64,
                }
            )

        # Convert Neo4j results to Polars DataFrame
        rows = []
        for result in results:
            product = result["product"]
            category = result.get("category", {})

            rows.append({
                "product_id": product["id"],
                "name": product["name"],
                "category": category.get("name") if category else None,
                "recent_purchases": result["recent_purchases"],
                "avg_price": result["avg_price"],
            })

        return pl.DataFrame(rows)

    async def get_category_popular(
        self,
        category_id: str,
        top_k: int = 10,
        min_purchases: int = 1,
    ) -> pl.DataFrame:
        """Get popular products in a category.

        Args:
            category_id: Category ID (e.g., "section_16", "garment_1010", "type_253")
            top_k: Number of products to return
            min_purchases: Minimum purchases to include product

        Returns:
            Polars DataFrame with columns: product_id, name, category, purchase_count, avg_price
        """
        results = await self.queries.get_category_recommendations(
            category_id=category_id,
            top_k=top_k,
            min_purchases=min_purchases,
        )

        if not results:
            return pl.DataFrame(
                schema={
                    "product_id": pl.Int64,
                    "name": pl.Utf8,
                    "category": pl.Utf8,
                    "purchase_count": pl.Int64,
                    "avg_price": pl.Float64,
                }
            )

        # Convert Neo4j results to Polars DataFrame
        rows = []
        for result in results:
            product = result["product"]
            category = result.get("category", {})

            rows.append({
                "product_id": product["id"],
                "name": product["name"],
                "category": category.get("name") if category else None,
                "purchase_count": result["purchase_count"],
                "avg_price": result.get("avg_price"),
            })

        return pl.DataFrame(rows)

    async def get_customer_journey(
        self,
        customer_id: str,
        limit: int = 50,
    ) -> pl.DataFrame:
        """Get customer's purchase journey over time.

        Args:
            customer_id: Customer ID
            limit: Maximum number of purchases to return

        Returns:
            Polars DataFrame with columns: product_id, name, category, timestamp, price
        """
        results = await self.queries.get_customer_journey(
            customer_id=customer_id,
            limit=limit,
        )

        if not results:
            return pl.DataFrame(
                schema={
                    "product_id": pl.Int64,
                    "name": pl.Utf8,
                    "category": pl.Utf8,
                    "timestamp": pl.Datetime,
                    "price": pl.Float64,
                }
            )

        # Convert Neo4j results to Polars DataFrame
        rows = []
        for result in results:
            product = result["product"]
            category = result.get("category", {})
            timestamp = result.get("timestamp")

            rows.append({
                "product_id": product["id"],
                "name": product["name"],
                "category": category.get("name") if category else None,
                "timestamp": timestamp,
                "price": result.get("price"),
            })

        return pl.DataFrame(rows)

    async def get_product_neighborhood(
        self,
        product_id: int,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get product's neighborhood in the graph.

        Returns products related through visual similarity, category, and co-purchases.

        Args:
            product_id: Article ID of the product
            depth: Maximum traversal depth (default: 2, max: 3)

        Returns:
            Dict with product info, categories, similar products, and co-purchased
        """
        return await self.queries.get_product_neighborhood(
            product_id=product_id,
            depth=depth,
        )

    async def get_product_stats(self, product_id: int) -> dict[str, Any]:
        """Get statistics for a product.

        Args:
            product_id: Article ID of the product

        Returns:
            Dict with purchase count, unique customers, avg price, etc.
        """
        return await self.queries.get_product_stats(product_id=product_id)
