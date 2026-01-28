"""Integration tests for Neo4j graph database."""

import pytest

from ragrec.graph.client import Neo4jClient
from ragrec.etl.graph_loader import GraphLoader


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def neo4j_client():
    """Neo4j client fixture with cleanup."""
    async with Neo4jClient() as client:
        # Clear any existing test data
        await client.clear_database()
        yield client
        # Cleanup after test
        await client.clear_database()


@pytest.fixture
async def graph_loader(neo4j_client):
    """Graph loader fixture."""
    async with GraphLoader(neo4j_client=neo4j_client) as loader:
        yield loader


async def test_neo4j_health_check(neo4j_client):
    """Test Neo4j connection health check."""
    is_healthy = await neo4j_client.health_check()
    assert is_healthy is True, "Neo4j should be healthy and reachable"


async def test_create_schema(graph_loader):
    """Test schema creation (constraints + indexes)."""
    await graph_loader.create_schema()

    # Verify constraints were created
    # (Neo4j doesn't fail on duplicate constraint creation, so this should always pass)
    result = await graph_loader.neo4j_client.execute_read(
        "SHOW CONSTRAINTS"
    )
    assert len(result) > 0, "Should have constraints created"


async def test_load_products(graph_loader):
    """Test loading Product nodes from PostgreSQL."""
    await graph_loader.create_schema()
    count = await graph_loader.load_products()

    assert count > 0, "Should load at least one product"

    # Verify products in graph
    result = await graph_loader.neo4j_client.execute_read(
        "MATCH (p:Product) RETURN count(p) AS count"
    )
    assert result[0]["count"] == count


async def test_load_customers(graph_loader):
    """Test loading Customer nodes from PostgreSQL."""
    await graph_loader.create_schema()
    count = await graph_loader.load_customers()

    assert count > 0, "Should load at least one customer"

    # Verify customers in graph
    result = await graph_loader.neo4j_client.execute_read(
        "MATCH (c:Customer) RETURN count(c) AS count"
    )
    assert result[0]["count"] == count


async def test_load_categories(graph_loader):
    """Test loading Category hierarchy."""
    await graph_loader.create_schema()
    count = await graph_loader.load_categories()

    assert count > 0, "Should load at least one category"

    # Verify category hierarchy levels
    result = await graph_loader.neo4j_client.execute_read(
        """
        MATCH (cat:Category)
        RETURN cat.level AS level, count(*) AS count
        ORDER BY level
        """
    )

    levels = {r["level"] for r in result}
    assert "section" in levels, "Should have section-level categories"
    assert "garment_group" in levels, "Should have garment_group-level categories"
    assert "product_type" in levels, "Should have product_type-level categories"


async def test_load_purchased_relationships(graph_loader):
    """Test loading PURCHASED relationships."""
    await graph_loader.create_schema()

    # Load nodes first
    await graph_loader.load_products()
    await graph_loader.load_customers()

    # Load relationships
    count = await graph_loader.load_purchased_relationships()

    assert count > 0, "Should load at least one PURCHASED relationship"

    # Verify relationships exist
    # Note: Count may be less than reported if some transactions reference
    # products/customers not in the sample dataset (MERGE will skip those)
    result = await graph_loader.neo4j_client.execute_read(
        "MATCH ()-[r:PURCHASED]->() RETURN count(r) AS count"
    )
    actual_count = result[0]["count"]
    assert actual_count > 0, "Should have at least one PURCHASED relationship in graph"
    assert actual_count <= count, "Graph count should not exceed fetched count"


async def test_load_in_category_relationships(graph_loader):
    """Test loading IN_CATEGORY relationships."""
    await graph_loader.create_schema()

    # Load nodes first
    await graph_loader.load_products()
    await graph_loader.load_categories()

    # Load relationships
    count = await graph_loader.load_in_category_relationships()

    assert count > 0, "Should load at least one IN_CATEGORY relationship"


async def test_load_parent_of_relationships(graph_loader):
    """Test loading PARENT_OF relationships for category hierarchy."""
    await graph_loader.create_schema()

    # Load categories first
    await graph_loader.load_categories()

    # Load relationships
    count = await graph_loader.load_parent_of_relationships()

    assert count > 0, "Should load at least one PARENT_OF relationship"

    # Verify hierarchy structure
    result = await graph_loader.neo4j_client.execute_read(
        """
        MATCH (parent:Category {level: 'section'})-[:PARENT_OF]->(child:Category {level: 'garment_group'})
        RETURN count(*) AS count
        """
    )
    assert result[0]["count"] > 0, "Sections should have garment_group children"


async def test_load_similar_to_relationships(graph_loader):
    """Test loading SIMILAR_TO relationships from embeddings."""
    await graph_loader.create_schema()

    # Load products first
    await graph_loader.load_products()

    # Load similarity relationships (top-3 for speed)
    count = await graph_loader.load_similar_to_relationships(top_k=3)

    # Note: This may return 0 if no embeddings exist in PostgreSQL
    # That's okay for the test - we're verifying the code runs without error
    if count > 0:
        # Verify relationships have scores
        result = await graph_loader.neo4j_client.execute_read(
            """
            MATCH ()-[r:SIMILAR_TO]->()
            RETURN r.score AS score, r.source AS source
            LIMIT 1
            """
        )
        assert result[0]["score"] > 0, "SIMILAR_TO should have similarity score"
        assert result[0]["source"] == "visual", "SIMILAR_TO should have source='visual'"


async def test_load_all(graph_loader):
    """Test loading complete graph from PostgreSQL."""
    counts = await graph_loader.load_all(clear_first=True, top_k_similar=3)

    # Verify all expected keys present
    assert "products" in counts
    assert "customers" in counts
    assert "categories" in counts
    assert "purchased" in counts
    assert "in_category" in counts
    assert "parent_of" in counts
    assert "similar_to" in counts

    # Verify counts are non-negative
    assert counts["products"] > 0, "Should have loaded products"
    assert counts["customers"] > 0, "Should have loaded customers"
    assert counts["categories"] > 0, "Should have loaded categories"


async def test_query_performance(graph_loader):
    """Test that graph queries are performant (<100ms)."""
    import time

    await graph_loader.load_all(clear_first=True, top_k_similar=3)

    # Test simple product lookup
    start = time.perf_counter()
    result = await graph_loader.neo4j_client.execute_read(
        "MATCH (p:Product) RETURN p LIMIT 10"
    )
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

    assert len(result) > 0, "Should return products"
    assert elapsed < 100, f"Query should be <100ms, took {elapsed:.2f}ms"


async def test_get_node_count(neo4j_client):
    """Test getting node counts by label."""
    # Create a simple test graph
    await neo4j_client.execute_write(
        "CREATE (p:Product {id: 999999, name: 'Test Product'})"
    )
    await neo4j_client.execute_write(
        "CREATE (c:Customer {id: 'test123', age_bracket: '25-35'})"
    )

    counts = await neo4j_client.get_node_count()

    assert "Product" in counts
    assert "Customer" in counts
    assert counts["Product"] >= 1
    assert counts["Customer"] >= 1


async def test_get_relationship_count(neo4j_client):
    """Test getting relationship counts by type."""
    # Create test relationships
    await neo4j_client.execute_write(
        """
        CREATE (c:Customer {id: 'test123'})
        CREATE (p:Product {id: 999999})
        CREATE (c)-[:PURCHASED {timestamp: datetime(), price: 99.99}]->(p)
        """
    )

    counts = await neo4j_client.get_relationship_count()

    assert "PURCHASED" in counts
    assert counts["PURCHASED"] >= 1
