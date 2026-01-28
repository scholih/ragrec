"""Load data from PostgreSQL into Neo4j graph."""

import asyncio
from typing import Any

import asyncpg
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ragrec.etl.config import ETLConfig
from ragrec.graph.client import Neo4jClient
from ragrec.graph import schema

console = Console()


class GraphLoader:
    """Bulk loader for Neo4j graph from PostgreSQL."""

    def __init__(
        self,
        pg_config: ETLConfig | None = None,
        neo4j_client: Neo4jClient | None = None,
    ) -> None:
        """Initialize graph loader.

        Args:
            pg_config: PostgreSQL configuration
            neo4j_client: Neo4j client (creates default if None)
        """
        self.pg_config = pg_config or ETLConfig()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self._owns_client = neo4j_client is None

    async def __aenter__(self) -> "GraphLoader":
        """Async context manager entry."""
        if self._owns_client:
            await self.neo4j_client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._owns_client:
            await self.neo4j_client.__aexit__(*args)

    async def create_schema(self) -> None:
        """Create constraints and indexes."""
        console.print("[bold blue]Creating schema (constraints + indexes)...[/bold blue]")

        for constraint in schema.CONSTRAINTS:
            try:
                await self.neo4j_client.execute_write(constraint)
                console.print(f"  ✓ {constraint.split()[1]}")
            except Exception as e:
                console.print(f"  [yellow]Skipped (may already exist): {e}[/yellow]")

        for index in schema.INDEXES:
            try:
                await self.neo4j_client.execute_write(index)
                console.print(f"  ✓ {index.split()[1]}")
            except Exception as e:
                console.print(f"  [yellow]Skipped (may already exist): {e}[/yellow]")

    async def load_products(self, batch_size: int = 1000) -> int:
        """Load Product nodes from PostgreSQL.

        Args:
            batch_size: Number of products per batch

        Returns:
            Total products loaded
        """
        console.print("[bold blue]Loading Product nodes...[/bold blue]")

        # Fetch products from PostgreSQL
        pool = await asyncpg.create_pool(self.pg_config.database_url)
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    article_id AS id,
                    prod_name AS name,
                    product_type_name AS product_type,
                    colour_group_name AS colour_group,
                    department_name AS department,
                    section_name AS section,
                    garment_group_name AS garment_group,
                    index_group_name AS index_group
                FROM products
            """)
        await pool.close()

        products = [dict(row) for row in rows]
        console.print(f"  Fetched {len(products)} products from PostgreSQL")

        # Batch import to Neo4j
        total = 0
        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            await self.neo4j_client.execute_write(
                schema.CREATE_PRODUCT_NODES,
                {"products": batch},
            )
            total += len(batch)

        console.print(f"  [green]✓ Loaded {total} Product nodes[/green]")
        return total

    async def load_customers(self, batch_size: int = 1000) -> int:
        """Load Customer nodes from PostgreSQL.

        Args:
            batch_size: Number of customers per batch

        Returns:
            Total customers loaded
        """
        console.print("[bold blue]Loading Customer nodes...[/bold blue]")

        # Fetch customers from PostgreSQL
        pool = await asyncpg.create_pool(self.pg_config.database_url)
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    customer_id AS id,
                    age_bracket,
                    club_member_status,
                    fashion_news_frequency
                FROM customers
            """)
        await pool.close()

        customers = [dict(row) for row in rows]
        console.print(f"  Fetched {len(customers)} customers from PostgreSQL")

        # Batch import to Neo4j
        total = 0
        for i in range(0, len(customers), batch_size):
            batch = customers[i : i + batch_size]
            await self.neo4j_client.execute_write(
                schema.CREATE_CUSTOMER_NODES,
                {"customers": batch},
            )
            total += len(batch)

        console.print(f"  [green]✓ Loaded {total} Customer nodes[/green]")
        return total

    async def load_categories(self) -> int:
        """Load Category nodes and hierarchy from H&M data.

        Creates 3-level hierarchy:
        - Level 1 (section): section_name
        - Level 2 (garment_group): garment_group_name
        - Level 3 (product_type): product_type_name

        Returns:
            Total categories loaded
        """
        console.print("[bold blue]Loading Category hierarchy...[/bold blue]")

        # Fetch unique categories from PostgreSQL
        pool = await asyncpg.create_pool(self.pg_config.database_url)
        async with pool.acquire() as conn:
            # Get unique sections (level 1)
            sections = await conn.fetch("""
                SELECT DISTINCT
                    section_no,
                    section_name
                FROM products
                WHERE section_name IS NOT NULL
            """)

            # Get unique garment groups (level 2) with parent section
            garment_groups = await conn.fetch("""
                SELECT DISTINCT
                    garment_group_no,
                    garment_group_name,
                    section_no,
                    section_name
                FROM products
                WHERE garment_group_name IS NOT NULL
            """)

            # Get unique product types (level 3) with parent garment group
            product_types = await conn.fetch("""
                SELECT DISTINCT
                    product_type_no,
                    product_type_name,
                    garment_group_no,
                    garment_group_name
                FROM products
                WHERE product_type_name IS NOT NULL
            """)
        await pool.close()

        categories = []

        # Level 1: Sections (no parent)
        for row in sections:
            categories.append({
                "id": f"section_{row['section_no']}",
                "name": row["section_name"],
                "level": "section",
                "parent_id": None,
            })

        # Level 2: Garment groups (parent = section)
        for row in garment_groups:
            categories.append({
                "id": f"garment_{row['garment_group_no']}",
                "name": row["garment_group_name"],
                "level": "garment_group",
                "parent_id": f"section_{row['section_no']}",
            })

        # Level 3: Product types (parent = garment group)
        for row in product_types:
            categories.append({
                "id": f"type_{row['product_type_no']}",
                "name": row["product_type_name"],
                "level": "product_type",
                "parent_id": f"garment_{row['garment_group_no']}",
            })

        console.print(f"  Built hierarchy: {len(categories)} total categories")
        console.print(f"    - Sections: {len(sections)}")
        console.print(f"    - Garment groups: {len(garment_groups)}")
        console.print(f"    - Product types: {len(product_types)}")

        # Import to Neo4j
        await self.neo4j_client.execute_write(
            schema.CREATE_CATEGORY_NODES,
            {"categories": categories},
        )

        console.print(f"  [green]✓ Loaded {len(categories)} Category nodes[/green]")
        return len(categories)

    async def load_purchased_relationships(self, batch_size: int = 1000) -> int:
        """Load PURCHASED relationships from transactions.

        Args:
            batch_size: Number of relationships per batch

        Returns:
            Total relationships created
        """
        console.print("[bold blue]Loading PURCHASED relationships...[/bold blue]")

        # Fetch transactions from PostgreSQL
        pool = await asyncpg.create_pool(self.pg_config.database_url)
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    customer_id,
                    article_id AS product_id,
                    transaction_date AS timestamp,
                    price
                FROM transactions
            """)
        await pool.close()

        purchases = [
            {
                "customer_id": row["customer_id"],
                "product_id": row["product_id"],
                "timestamp": row["timestamp"].isoformat(),
                "price": float(row["price"]),
            }
            for row in rows
        ]

        console.print(f"  Fetched {len(purchases)} transactions from PostgreSQL")

        # Batch import to Neo4j
        total = 0
        for i in range(0, len(purchases), batch_size):
            batch = purchases[i : i + batch_size]
            await self.neo4j_client.execute_write(
                schema.CREATE_PURCHASED_RELATIONSHIPS,
                {"purchases": batch},
            )
            total += len(batch)

        console.print(f"  [green]✓ Loaded {total} PURCHASED relationships[/green]")
        return total

    async def load_in_category_relationships(self) -> int:
        """Load IN_CATEGORY relationships linking products to categories.

        Returns:
            Total relationships created
        """
        console.print("[bold blue]Loading IN_CATEGORY relationships...[/bold blue]")

        result = await self.neo4j_client.execute_write(schema.CREATE_IN_CATEGORY_RELATIONSHIPS)

        # Count relationships created
        count_result = await self.neo4j_client.execute_read(
            "MATCH ()-[r:IN_CATEGORY]->() RETURN count(r) AS count"
        )
        total = count_result[0]["count"] if count_result else 0

        console.print(f"  [green]✓ Loaded {total} IN_CATEGORY relationships[/green]")
        return total

    async def load_parent_of_relationships(self) -> int:
        """Load PARENT_OF relationships for category hierarchy.

        Returns:
            Total relationships created
        """
        console.print("[bold blue]Loading PARENT_OF relationships...[/bold blue]")

        result = await self.neo4j_client.execute_write(schema.CREATE_PARENT_OF_RELATIONSHIPS)

        # Count relationships created
        count_result = await self.neo4j_client.execute_read(
            "MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS count"
        )
        total = count_result[0]["count"] if count_result else 0

        console.print(f"  [green]✓ Loaded {total} PARENT_OF relationships[/green]")
        return total

    async def load_similar_to_relationships(
        self,
        top_k: int = 5,
        batch_size: int = 1000,
    ) -> int:
        """Load SIMILAR_TO relationships from visual similarity.

        Queries PostgreSQL for top-k most similar products for each product
        based on embedding cosine distance.

        Args:
            top_k: Number of similar products per product
            batch_size: Number of relationships per batch

        Returns:
            Total relationships created
        """
        console.print(f"[bold blue]Loading SIMILAR_TO relationships (top-{top_k})...[/bold blue]")

        # Fetch visual similarities from PostgreSQL
        pool = await asyncpg.create_pool(self.pg_config.database_url)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH similarities AS (
                    SELECT
                        pe1.article_id AS product_id,
                        pe2.article_id AS similar_id,
                        pe1.embedding <=> pe2.embedding AS distance
                    FROM product_embeddings pe1
                    CROSS JOIN product_embeddings pe2
                    WHERE pe1.article_id != pe2.article_id
                ),
                ranked AS (
                    SELECT
                        product_id,
                        similar_id,
                        distance,
                        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY distance) AS rank
                    FROM similarities
                )
                SELECT
                    product_id,
                    similar_id,
                    distance
                FROM ranked
                WHERE rank <= {top_k}
                ORDER BY product_id, distance
                """
            )
        await pool.close()

        similarities = [
            {
                "product_id": row["product_id"],
                "similar_id": row["similar_id"],
                "score": 1.0 - float(row["distance"]),  # Convert distance to similarity score
            }
            for row in rows
        ]

        console.print(f"  Computed {len(similarities)} similarities from embeddings")

        # Batch import to Neo4j
        total = 0
        for i in range(0, len(similarities), batch_size):
            batch = similarities[i : i + batch_size]
            await self.neo4j_client.execute_write(
                schema.CREATE_SIMILAR_TO_RELATIONSHIPS,
                {"similarities": batch},
            )
            total += len(batch)

        console.print(f"  [green]✓ Loaded {total} SIMILAR_TO relationships[/green]")
        return total

    async def load_all(
        self,
        clear_first: bool = False,
        top_k_similar: int = 5,
    ) -> dict[str, int]:
        """Load complete graph from PostgreSQL.

        Args:
            clear_first: If True, clear existing data before loading
            top_k_similar: Number of similar products per product

        Returns:
            Dict with counts of loaded entities
        """
        if clear_first:
            console.print("[bold yellow]Clearing existing graph data...[/bold yellow]")
            await self.neo4j_client.clear_database()

        counts = {}

        # Step 1: Create schema
        await self.create_schema()

        # Step 2: Load nodes
        counts["products"] = await self.load_products()
        counts["customers"] = await self.load_customers()
        counts["categories"] = await self.load_categories()

        # Step 3: Load relationships
        counts["purchased"] = await self.load_purchased_relationships()
        counts["in_category"] = await self.load_in_category_relationships()
        counts["parent_of"] = await self.load_parent_of_relationships()
        counts["similar_to"] = await self.load_similar_to_relationships(top_k=top_k_similar)

        console.print("\n[bold green]✓ Graph load complete![/bold green]")
        console.print("\nSummary:")
        for key, value in counts.items():
            console.print(f"  {key}: {value:,}")

        return counts


async def load_graph_from_postgres(
    clear_first: bool = False,
    top_k_similar: int = 5,
) -> dict[str, int]:
    """Convenience function to load graph from PostgreSQL.

    Args:
        clear_first: If True, clear existing data before loading
        top_k_similar: Number of similar products per product

    Returns:
        Dict with counts of loaded entities
    """
    async with GraphLoader() as loader:
        return await loader.load_all(clear_first=clear_first, top_k_similar=top_k_similar)


if __name__ == "__main__":
    # CLI usage
    asyncio.run(load_graph_from_postgres(clear_first=True, top_k_similar=5))
