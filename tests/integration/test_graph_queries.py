"""Integration tests for graph queries and collaborative recommender."""

import pytest

from ragrec.graph.client import Neo4jClient
from ragrec.graph.queries import GraphQueries
from ragrec.recommender.collaborative import CollaborativeRecommender
from ragrec.etl.graph_loader import GraphLoader


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def loaded_graph():
    """Load sample data into Neo4j for testing."""
    async with GraphLoader() as loader:
        # Load sample data
        await loader.load_all(clear_first=True, top_k_similar=3)
        yield loader.neo4j_client


@pytest.fixture
async def graph_queries(loaded_graph):
    """GraphQueries instance with loaded data."""
    return GraphQueries(loaded_graph)


@pytest.fixture
async def collaborative_recommender(loaded_graph):
    """CollaborativeRecommender instance with loaded data."""
    async with CollaborativeRecommender(neo4j_client=loaded_graph) as recommender:
        yield recommender


async def test_get_product_neighborhood(graph_queries):
    """Test getting product neighborhood."""
    # Get first product ID from graph
    client = graph_queries.client
    result = await client.execute_read("MATCH (p:Product) RETURN p.id AS id LIMIT 1")
    product_id = result[0]["id"]

    neighborhood = await graph_queries.get_product_neighborhood(
        product_id=product_id,
        depth=2,
    )

    assert neighborhood is not None
    assert "product" in neighborhood
    assert neighborhood["product"]["id"] == product_id
    assert "categories" in neighborhood
    assert "similar_products" in neighborhood


async def test_get_customer_journey(graph_queries):
    """Test getting customer purchase journey."""
    # Get first customer ID from graph
    client = graph_queries.client
    result = await client.execute_read("MATCH (c:Customer) RETURN c.id AS id LIMIT 1")
    customer_id = result[0]["id"]

    journey = await graph_queries.get_customer_journey(
        customer_id=customer_id,
        limit=10,
    )

    assert isinstance(journey, list)
    # Customer may or may not have purchases in sample data
    if journey:
        assert "product" in journey[0]
        assert "timestamp" in journey[0]


async def test_get_collaborative_recommendations(graph_queries):
    """Test collaborative filtering recommendations."""
    # Get customer with purchases
    client = graph_queries.client
    result = await client.execute_read(
        """
        MATCH (c:Customer)-[:PURCHASED]->(:Product)
        RETURN c.id AS id
        LIMIT 1
        """
    )

    if not result:
        pytest.skip("No customers with purchases in test data")

    customer_id = result[0]["id"]

    recommendations = await graph_queries.get_collaborative_recommendations(
        customer_id=customer_id,
        top_k=5,
        min_shared_purchases=1,
    )

    assert isinstance(recommendations, list)
    # May be empty if no similar customers
    if recommendations:
        assert "product" in recommendations[0]
        assert "score" in recommendations[0]


async def test_find_complementary_products(graph_queries):
    """Test finding complementary products."""
    # Get product that has been purchased
    client = graph_queries.client
    result = await client.execute_read(
        """
        MATCH (p:Product)<-[:PURCHASED]-(:Customer)
        RETURN p.id AS id
        LIMIT 1
        """
    )

    if not result:
        pytest.skip("No products with purchases in test data")

    product_id = result[0]["id"]

    complementary = await graph_queries.find_complementary_products(
        product_id=product_id,
        top_k=5,
        min_co_purchases=1,
    )

    assert isinstance(complementary, list)
    # May be empty if no co-purchases
    if complementary:
        assert "product" in complementary[0]
        assert "co_purchase_count" in complementary[0]


async def test_get_trending_products(graph_queries):
    """Test getting trending products."""
    trending = await graph_queries.get_trending_products(
        days=30,  # Use wider window for test data
        top_k=5,
    )

    assert isinstance(trending, list)
    # Should have trending products if we have purchases
    if trending:
        assert "product" in trending[0]
        assert "recent_purchases" in trending[0]


async def test_get_product_stats(graph_queries):
    """Test getting product statistics."""
    # Get first product
    client = graph_queries.client
    result = await client.execute_read("MATCH (p:Product) RETURN p.id AS id LIMIT 1")
    product_id = result[0]["id"]

    stats = await graph_queries.get_product_stats(product_id=product_id)

    assert stats is not None
    assert "product" in stats
    assert "total_purchases" in stats
    assert "unique_customers" in stats


async def test_collaborative_recommender_for_customer(collaborative_recommender):
    """Test collaborative recommender."""
    # Get customer with purchases
    client = collaborative_recommender.neo4j_client
    result = await client.execute_read(
        """
        MATCH (c:Customer)-[:PURCHASED]->(:Product)
        RETURN c.id AS id
        LIMIT 1
        """
    )

    if not result:
        pytest.skip("No customers with purchases in test data")

    customer_id = result[0]["id"]

    recommendations_df = await collaborative_recommender.recommend_for_customer(
        customer_id=customer_id,
        top_k=5,
        min_shared_purchases=1,
    )

    assert recommendations_df is not None
    assert "product_id" in recommendations_df.columns
    assert "name" in recommendations_df.columns
    assert "score" in recommendations_df.columns


async def test_collaborative_recommender_complementary(collaborative_recommender):
    """Test complementary product recommendations."""
    # Get product with purchases
    client = collaborative_recommender.neo4j_client
    result = await client.execute_read(
        """
        MATCH (p:Product)<-[:PURCHASED]-(:Customer)
        RETURN p.id AS id
        LIMIT 1
        """
    )

    if not result:
        pytest.skip("No products with purchases in test data")

    product_id = result[0]["id"]

    complementary_df = await collaborative_recommender.recommend_complementary(
        product_id=product_id,
        top_k=5,
        min_co_purchases=1,
    )

    assert complementary_df is not None
    assert "product_id" in complementary_df.columns
    assert "name" in complementary_df.columns
    assert "co_purchase_count" in complementary_df.columns


async def test_collaborative_recommender_trending(collaborative_recommender):
    """Test trending products."""
    trending_df = await collaborative_recommender.get_trending(
        days=30,
        top_k=5,
    )

    assert trending_df is not None
    assert "product_id" in trending_df.columns
    assert "name" in trending_df.columns
    assert "recent_purchases" in trending_df.columns


async def test_collaborative_recommender_customer_journey(collaborative_recommender):
    """Test customer journey."""
    # Get customer with purchases
    client = collaborative_recommender.neo4j_client
    result = await client.execute_read(
        """
        MATCH (c:Customer)-[:PURCHASED]->(:Product)
        RETURN c.id AS id
        LIMIT 1
        """
    )

    if not result:
        pytest.skip("No customers with purchases in test data")

    customer_id = result[0]["id"]

    journey_df = await collaborative_recommender.get_customer_journey(
        customer_id=customer_id,
        limit=10,
    )

    assert journey_df is not None
    assert "product_id" in journey_df.columns
    assert "name" in journey_df.columns
    assert "timestamp" in journey_df.columns


async def test_collaborative_recommender_product_neighborhood(collaborative_recommender):
    """Test product neighborhood."""
    # Get first product
    client = collaborative_recommender.neo4j_client
    result = await client.execute_read("MATCH (p:Product) RETURN p.id AS id LIMIT 1")
    product_id = result[0]["id"]

    neighborhood = await collaborative_recommender.get_product_neighborhood(
        product_id=product_id,
        depth=2,
    )

    assert neighborhood is not None
    assert "product" in neighborhood
    assert neighborhood["product"]["id"] == product_id


async def test_category_recommendations(graph_queries):
    """Test category-based recommendations."""
    # Get first category
    client = graph_queries.client
    result = await client.execute_read(
        """
        MATCH (cat:Category)
        WHERE cat.level = 'product_type'
        RETURN cat.id AS id
        LIMIT 1
        """
    )

    if not result:
        pytest.skip("No categories in test data")

    category_id = result[0]["id"]

    recommendations = await graph_queries.get_category_recommendations(
        category_id=category_id,
        top_k=5,
        min_purchases=0,
    )

    assert isinstance(recommendations, list)
    # Should have products in category
    if recommendations:
        assert "product" in recommendations[0]
        assert "purchase_count" in recommendations[0]


async def test_collaborative_recommender_category_popular(collaborative_recommender):
    """Test category popular products."""
    # Get first category
    client = collaborative_recommender.neo4j_client
    result = await client.execute_read(
        """
        MATCH (cat:Category)
        WHERE cat.level = 'product_type'
        RETURN cat.id AS id
        LIMIT 1
        """
    )

    if not result:
        pytest.skip("No categories in test data")

    category_id = result[0]["id"]

    popular_df = await collaborative_recommender.get_category_popular(
        category_id=category_id,
        top_k=5,
        min_purchases=0,
    )

    assert popular_df is not None
    assert "product_id" in popular_df.columns
    assert "name" in popular_df.columns
    assert "purchase_count" in popular_df.columns
