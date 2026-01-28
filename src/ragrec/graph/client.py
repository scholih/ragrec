"""Async Neo4j client."""

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jConfig(BaseSettings):
    """Neo4j connection configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ragrec123"  # Change in production!


class Neo4jClient:
    """Async Neo4j graph database client."""

    def __init__(self, config: Neo4jConfig | None = None) -> None:
        """Initialize Neo4j client.

        Args:
            config: Neo4j configuration (uses defaults if None)
        """
        self.config = config or Neo4jConfig()
        self.driver: AsyncDriver | None = None

    async def __aenter__(self) -> "Neo4jClient":
        """Async context manager entry."""
        self.driver = AsyncGraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self.driver:
            await self.driver.close()

    async def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a write query (CREATE, MERGE, etc.).

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized. Use async context manager.")

        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query (MATCH, RETURN, etc.).

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized. Use async context manager.")

        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def health_check(self) -> bool:
        """Check if Neo4j is reachable and healthy.

        Returns:
            True if healthy, False otherwise
        """
        if not self.driver:
            return False

        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 AS health")
                data = await result.data()
                return len(data) > 0 and data[0].get("health") == 1
        except Exception:
            return False

    async def clear_database(self) -> None:
        """Clear all nodes and relationships (use with caution!).

        This is useful for testing and development.
        """
        await self.execute_write("MATCH (n) DETACH DELETE n")

    async def get_node_count(self) -> dict[str, int]:
        """Get count of nodes by label.

        Returns:
            Dict mapping label names to counts
        """
        query = """
        CALL db.labels() YIELD label
        CALL {
            WITH label
            MATCH (n)
            WHERE label IN labels(n)
            RETURN count(n) AS count
        }
        RETURN label, count
        ORDER BY count DESC
        """

        results = await self.execute_read(query)
        return {record["label"]: record["count"] for record in results}

    async def get_relationship_count(self) -> dict[str, int]:
        """Get count of relationships by type.

        Returns:
            Dict mapping relationship types to counts
        """
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL {
            WITH relationshipType
            MATCH ()-[r]->()
            WHERE type(r) = relationshipType
            RETURN count(r) AS count
        }
        RETURN relationshipType, count
        ORDER BY count DESC
        """

        results = await self.execute_read(query)
        return {record["relationshipType"]: record["count"] for record in results}
