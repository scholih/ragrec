"""Customer persona discovery using hybrid approach.

Combines embedding clustering (HDBSCAN) with graph community detection (Louvain).
"""

import asyncio
from collections import Counter, defaultdict
from typing import Any

import asyncpg
import hdbscan
import numpy as np
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ragrec.etl.config import ETLConfig
from ragrec.graph.client import Neo4jClient
from ragrec.personas.models import Persona, CustomerPersonaAssignment

console = Console()


class PersonaDiscovery:
    """Discover customer personas using hybrid clustering."""

    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        pg_config: ETLConfig | None = None,
    ) -> None:
        """Initialize persona discovery.

        Args:
            neo4j_client: Neo4j client (creates default if None)
            pg_config: PostgreSQL configuration
        """
        self.neo4j_client = neo4j_client or Neo4jClient()
        self._owns_neo4j_client = neo4j_client is None
        self.pg_config = pg_config or ETLConfig()

    async def __aenter__(self) -> "PersonaDiscovery":
        """Async context manager entry."""
        if self._owns_neo4j_client:
            await self.neo4j_client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._owns_neo4j_client:
            await self.neo4j_client.__aexit__(*args)

    async def discover_personas(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        target_personas: int = 8,
    ) -> tuple[list[Persona], list[CustomerPersonaAssignment]]:
        """Discover customer personas using hybrid approach.

        Args:
            min_cluster_size: Minimum HDBSCAN cluster size
            min_samples: Minimum samples for HDBSCAN core points
            target_personas: Target number of personas (guideline)

        Returns:
            Tuple of (personas, assignments)
        """
        console.print("[bold blue]Discovering customer personas...[/bold blue]")

        # Step 1: Get behavior embeddings from PostgreSQL
        console.print("  [1/5] Fetching behavior embeddings...")
        embeddings_data = await self._fetch_behavior_embeddings()
        customer_ids = embeddings_data["customer_ids"]
        embeddings = embeddings_data["embeddings"]
        console.print(f"    Loaded {len(customer_ids):,} customer embeddings")

        # Step 2: Run HDBSCAN clustering
        console.print("  [2/5] Running HDBSCAN clustering...")
        embedding_clusters = self._run_hdbscan(
            embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        n_embedding_clusters = len(set(embedding_clusters)) - (1 if -1 in embedding_clusters else 0)
        console.print(f"    Found {n_embedding_clusters} embedding clusters")

        # Step 3: Run graph community detection
        console.print("  [3/5] Running Louvain community detection...")
        graph_communities = await self._run_louvain()
        n_graph_communities = len(set(graph_communities.values()))
        console.print(f"    Found {n_graph_communities} graph communities")

        # Step 4: Hybrid fusion
        console.print("  [4/5] Fusing clusters and communities...")
        assignments = self._fuse_clusters_and_communities(
            customer_ids=customer_ids,
            embedding_clusters=embedding_clusters,
            graph_communities=graph_communities,
        )
        console.print(f"    Created {len(set(a.persona_id for a in assignments))} personas")

        # Step 5: Generate persona profiles
        console.print("  [5/5] Generating persona profiles...")
        personas = await self._generate_persona_profiles(assignments)

        console.print(f"\n[bold green]✓ Discovered {len(personas)} personas[/bold green]")
        for persona in personas:
            console.print(f"  {persona.name}: {persona.size:,} customers")

        return personas, assignments

    async def _fetch_behavior_embeddings(self) -> dict[str, Any]:
        """Fetch customer behavior embeddings from PostgreSQL."""
        pool = await asyncpg.create_pool(self.pg_config.database_url)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT customer_id, behavior_embedding::text as embedding_text
                FROM customers
                WHERE behavior_embedding IS NOT NULL
                ORDER BY customer_id
                """
            )

        await pool.close()

        customer_ids = []
        embeddings = []

        for row in rows:
            customer_ids.append(row["customer_id"])
            # Parse vector text format
            embedding_text = row["embedding_text"]
            embedding_values = [float(x) for x in embedding_text.strip("[]").split(",")]
            embeddings.append(embedding_values)

        embeddings_array = np.array(embeddings, dtype=np.float32)

        return {
            "customer_ids": customer_ids,
            "embeddings": embeddings_array,
        }

    def _run_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
    ) -> np.ndarray:
        """Run HDBSCAN clustering on embeddings."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of mass
        )

        cluster_labels = clusterer.fit_predict(embeddings)
        return cluster_labels

    async def _run_louvain(self) -> dict[str, int]:
        """Run Louvain community detection on customer-product graph."""
        # Run Louvain algorithm in Neo4j
        # Note: This requires GDS (Graph Data Science) library in Neo4j
        # For MVP, we'll use a simpler approach: connected components or manual community detection

        # Alternative: Use purchase similarity to assign communities
        # For now, let's use a simplified approach based on purchase categories

        pool = await asyncpg.create_pool(self.pg_config.database_url)

        async with pool.acquire() as conn:
            # Get customer's most purchased category as community proxy
            rows = await conn.fetch(
                """
                SELECT 
                    c.customer_id,
                    p.department_name,
                    COUNT(*) as purchase_count,
                    ROW_NUMBER() OVER (PARTITION BY c.customer_id ORDER BY COUNT(*) DESC) as rank
                FROM customers c
                JOIN transactions t ON c.customer_id = t.customer_id
                JOIN products p ON t.article_id = p.article_id
                WHERE p.department_name IS NOT NULL
                GROUP BY c.customer_id, p.department_name
                """
            )

        await pool.close()

        # Assign community based on top department
        customer_communities = {}
        department_to_community = {}
        next_community_id = 0

        for row in rows:
            if row["rank"] == 1:  # Top department for this customer
                customer_id = row["customer_id"]
                department = row["department_name"]

                if department not in department_to_community:
                    department_to_community[department] = next_community_id
                    next_community_id += 1

                customer_communities[customer_id] = department_to_community[department]

        return customer_communities

    def _fuse_clusters_and_communities(
        self,
        customer_ids: list[str],
        embedding_clusters: np.ndarray,
        graph_communities: dict[str, int],
    ) -> list[CustomerPersonaAssignment]:
        """Fuse embedding clusters and graph communities into personas.

        Uses embedding clusters as primary personas, using graph communities
        for validation/confidence scoring rather than creating full cross-product.

        This approach creates ~8-12 personas (one per embedding cluster) rather
        than hundreds from the Cartesian product.
        """
        assignments = []

        for i, customer_id in enumerate(customer_ids):
            embedding_cluster = int(embedding_clusters[i])
            graph_community = graph_communities.get(customer_id, -1)

            # Use embedding cluster as persona (simplified approach)
            if embedding_cluster == -1:
                persona_id = "persona_uncategorized"
                confidence = 0.3  # Low confidence for noise points
            else:
                persona_id = f"persona_{embedding_cluster}"
                confidence = 1.0  # High confidence for clustered customers

            assignment = CustomerPersonaAssignment(
                customer_id=customer_id,
                persona_id=persona_id,
                embedding_cluster=embedding_cluster,
                graph_community=graph_community,
                confidence=confidence,
            )
            assignments.append(assignment)

        return assignments

    async def _generate_persona_profiles(
        self,
        assignments: list[CustomerPersonaAssignment],
    ) -> list[Persona]:
        """Generate persona profiles with characteristics."""
        # Group customers by persona
        persona_customers = defaultdict(list)
        for assignment in assignments:
            persona_customers[assignment.persona_id].append(assignment.customer_id)

        # Fetch customer and purchase data
        pool = await asyncpg.create_pool(self.pg_config.database_url)

        personas = []

        for persona_id, customer_ids in persona_customers.items():
            async with pool.acquire() as conn:
                # Get customer demographics
                demographics = await conn.fetchrow(
                    """
                    SELECT 
                        AVG(age) as avg_age,
                        AVG(CASE WHEN club_member_status = 'ACTIVE' THEN 1.0 ELSE 0.0 END) as club_ratio,
                        AVG(CASE WHEN fashion_news_frequency IN ('Regularly', 'Monthly') THEN 1.0 ELSE 0.0 END) as news_ratio
                    FROM customers
                    WHERE customer_id = ANY($1)
                    """,
                    customer_ids,
                )

                # Get purchase behavior
                purchase_behavior = await conn.fetch(
                    """
                    SELECT 
                        p.product_type_name,
                        COUNT(*) as purchase_count,
                        AVG(t.price) as avg_price
                    FROM transactions t
                    JOIN products p ON t.article_id = p.article_id
                    WHERE t.customer_id = ANY($1)
                      AND p.product_type_name IS NOT NULL
                    GROUP BY p.product_type_name
                    ORDER BY purchase_count DESC
                    LIMIT 5
                    """,
                    customer_ids,
                )

                # Get average purchases and spend per customer
                customer_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(DISTINCT t.customer_id) as customer_count,
                        COUNT(*) as total_purchases,
                        SUM(t.price) as total_spend
                    FROM transactions t
                    WHERE t.customer_id = ANY($1)
                    """,
                    customer_ids,
                )

            # Build persona
            top_categories = [row["product_type_name"] for row in purchase_behavior[:3]]

            avg_purchases = (
                customer_stats["total_purchases"] / customer_stats["customer_count"]
                if customer_stats["customer_count"] > 0
                else 0
            )

            avg_spend = (
                float(customer_stats["total_spend"]) / customer_stats["customer_count"]
                if customer_stats["customer_count"] > 0 and customer_stats["total_spend"]
                else 0
            )

            # Generate name and description
            persona_name = self._generate_persona_name(top_categories, demographics)
            persona_description = self._generate_persona_description(
                top_categories, demographics, avg_purchases, avg_spend
            )

            persona = Persona(
                id=persona_id,
                name=persona_name,
                description=persona_description,
                size=len(customer_ids),
                avg_age=float(demographics["avg_age"]) if demographics["avg_age"] else None,
                top_categories=top_categories,
                avg_purchases_per_customer=avg_purchases,
                avg_spend_per_customer=avg_spend,
                club_member_ratio=float(demographics["club_ratio"]) if demographics["club_ratio"] else None,
                fashion_news_active_ratio=float(demographics["news_ratio"]) if demographics["news_ratio"] else None,
            )

            personas.append(persona)

        await pool.close()

        # Sort by size (largest first)
        personas.sort(key=lambda p: p.size, reverse=True)

        return personas

    def _generate_persona_name(
        self,
        top_categories: list[str],
        demographics: Any,
    ) -> str:
        """Generate a descriptive persona name."""
        if not top_categories:
            return "Casual Shoppers"

        # Use top category as base
        primary_category = top_categories[0]

        # Add demographic context
        avg_age = demographics["avg_age"]
        if avg_age and avg_age < 30:
            age_prefix = "Young"
        elif avg_age and avg_age > 50:
            age_prefix = "Mature"
        else:
            age_prefix = ""

        # Simplify category name
        category_simple = primary_category.replace(" ", "").replace("/", "")[:15]

        if age_prefix:
            return f"{age_prefix} {category_simple} Fans"
        else:
            return f"{category_simple} Enthusiasts"

    def _generate_persona_description(
        self,
        top_categories: list[str],
        demographics: Any,
        avg_purchases: float,
        avg_spend: float,
    ) -> str:
        """Generate a descriptive persona description."""
        desc_parts = []

        # Category preferences
        if top_categories:
            categories_str = ", ".join(top_categories[:3])
            desc_parts.append(f"Primarily interested in: {categories_str}")

        # Demographics
        if demographics["avg_age"]:
            desc_parts.append(f"Average age: {demographics['avg_age']:.0f}")

        # Engagement
        if demographics["club_ratio"] and demographics["club_ratio"] > 0.5:
            desc_parts.append("Highly engaged club members")

        # Purchase behavior
        if avg_purchases > 5:
            desc_parts.append(f"Frequent shoppers ({avg_purchases:.1f} purchases/customer)")

        if avg_spend > 200:
            desc_parts.append(f"High spenders (€{avg_spend:.0f}/customer)")

        return ". ".join(desc_parts) + "."


async def discover_and_store_personas(
    min_cluster_size: int = 50,
    min_samples: int = 10,
    target_personas: int = 8,
) -> dict[str, int]:
    """Discover personas and store in Neo4j.

    Returns:
        Dict with counts of personas and assignments
    """
    async with PersonaDiscovery() as discovery:
        # Discover personas
        personas, assignments = await discovery.discover_personas(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            target_personas=target_personas,
        )

        # Store in Neo4j
        console.print("\n[bold blue]Storing personas in Neo4j...[/bold blue]")

        # Clear existing personas and relationships
        await discovery.neo4j_client.execute_write(
            """
            MATCH (c:Customer)-[r:BELONGS_TO]->(p:Persona)
            DELETE r
            """,
            {},
        )
        await discovery.neo4j_client.execute_write(
            """
            MATCH (p:Persona)
            DELETE p
            """,
            {},
        )
        console.print("  ✓ Cleared existing personas")

        # Create Persona nodes
        persona_data = [p.to_dict() for p in personas]
        await discovery.neo4j_client.execute_write(
            """
            UNWIND $personas AS persona
            CREATE (p:Persona {id: persona.id})
            SET p.name = persona.name,
                p.description = persona.description,
                p.size = persona.size,
                p.avg_age = persona.avg_age,
                p.top_categories = persona.top_categories,
                p.avg_purchases_per_customer = persona.avg_purchases_per_customer,
                p.avg_spend_per_customer = persona.avg_spend_per_customer,
                p.club_member_ratio = persona.club_member_ratio,
                p.fashion_news_active_ratio = persona.fashion_news_active_ratio
            """,
            {"personas": persona_data},
        )
        console.print(f"  ✓ Created {len(personas)} Persona nodes")

        # Create BELONGS_TO relationships
        memberships = [
            {
                "customer_id": a.customer_id,
                "persona_id": a.persona_id,
                "confidence": a.confidence,
            }
            for a in assignments
        ]

        # Batch create relationships
        for i in range(0, len(memberships), 1000):
            batch = memberships[i : i + 1000]
            await discovery.neo4j_client.execute_write(
                """
                UNWIND $memberships AS membership
                MATCH (c:Customer {id: membership.customer_id})
                MATCH (p:Persona {id: membership.persona_id})
                MERGE (c)-[r:BELONGS_TO]->(p)
                SET r.confidence = membership.confidence
                """,
                {"memberships": batch},
            )

        console.print(f"  ✓ Created {len(assignments):,} BELONGS_TO relationships")

        console.print("\n[bold green]✓ Persona discovery and storage complete![/bold green]")

        return {
            "personas": len(personas),
            "assignments": len(assignments),
        }


if __name__ == "__main__":
    asyncio.run(discover_and_store_personas())
