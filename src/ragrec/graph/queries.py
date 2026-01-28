"""Graph query patterns for recommendations."""

from typing import Any

from ragrec.graph.client import Neo4jClient


class GraphQueries:
    """Common graph query patterns for recommendations."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize graph queries.

        Args:
            client: Neo4j client instance
        """
        self.client = client

    async def get_product_neighborhood(
        self,
        product_id: int,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get product's neighborhood in the graph.

        Returns products related through:
        - Visual similarity (SIMILAR_TO edges)
        - Same category (IN_CATEGORY -> Category <- IN_CATEGORY)
        - Co-purchased (Customer -> PURCHASED -> Product)

        Args:
            product_id: Article ID of the product
            depth: Maximum traversal depth (default: 2, max: 3)

        Returns:
            Dict with product info, categories, similar products, and co-purchased
        """
        # Limit depth to prevent graph explosion
        depth = min(depth, 3)

        query = """
        MATCH (p:Product {id: $product_id})
        
        // Get categories
        OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
        WITH p, collect(DISTINCT cat) AS categories
        
        // Get visually similar products
        OPTIONAL MATCH (p)-[sim:SIMILAR_TO]->(similar:Product)
        WITH p, categories, collect(DISTINCT {
            product: similar,
            score: sim.score,
            source: sim.source
        })[..10] AS similar_products
        
        // Get co-purchased products (customers who bought this also bought...)
        OPTIONAL MATCH (customer:Customer)-[:PURCHASED]->(p)
        OPTIONAL MATCH (customer)-[:PURCHASED]->(co_purchased:Product)
        WHERE co_purchased.id <> p.id
        WITH p, categories, similar_products, 
             co_purchased, count(DISTINCT customer) AS co_purchase_count
        WITH p, categories, similar_products,
             collect({
                 product: co_purchased,
                 count: co_purchase_count
             })[..10] AS co_purchased
        
        // Get products in same category
        OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat)-[:PARENT_OF*0..1]->(:Category)<-[:IN_CATEGORY]-(same_cat:Product)
        WHERE same_cat.id <> p.id
        
        RETURN 
            p AS product,
            categories,
            similar_products,
            co_purchased,
            collect(DISTINCT same_cat)[..10] AS same_category
        """

        results = await self.client.execute_read(query, {"product_id": product_id})

        if not results:
            return {}

        return results[0]

    async def get_customer_journey(
        self,
        customer_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get customer's purchase journey over time.

        Args:
            customer_id: Customer ID
            limit: Maximum number of purchases to return

        Returns:
            List of purchases with product details, ordered by timestamp
        """
        query = """
        MATCH (c:Customer {id: $customer_id})-[pur:PURCHASED]->(p:Product)
        OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
        
        RETURN 
            p AS product,
            cat AS category,
            pur.timestamp AS timestamp,
            pur.price AS price
        ORDER BY pur.timestamp DESC
        LIMIT $limit
        """

        results = await self.client.execute_read(
            query,
            {"customer_id": customer_id, "limit": limit},
        )

        return results

    async def get_collaborative_recommendations(
        self,
        customer_id: str,
        top_k: int = 10,
        min_shared_purchases: int = 2,
    ) -> list[dict[str, Any]]:
        """Get collaborative filtering recommendations.

        Finds products purchased by customers with similar purchase history.

        Args:
            customer_id: Customer ID
            top_k: Number of recommendations to return
            min_shared_purchases: Minimum shared purchases to consider customers similar

        Returns:
            List of recommended products with scores
        """
        query = """
        // Find target customer's purchases
        MATCH (target:Customer {id: $customer_id})-[:PURCHASED]->(p:Product)
        WITH target, collect(p.id) AS target_products
        
        // Find similar customers (shared at least N purchases)
        MATCH (similar:Customer)-[:PURCHASED]->(shared:Product)
        WHERE shared.id IN target_products
          AND similar.id <> target.id
        WITH target, target_products, similar, count(DISTINCT shared) AS shared_count
        WHERE shared_count >= $min_shared_purchases
        
        // Get products purchased by similar customers but not by target
        MATCH (similar)-[:PURCHASED]->(rec:Product)
        WHERE NOT rec.id IN target_products
        
        // Count how many similar customers bought each product (popularity signal)
        WITH rec, count(DISTINCT similar) AS similar_customer_count
        
        // Get additional product info
        OPTIONAL MATCH (rec)-[:IN_CATEGORY]->(cat:Category)
        
        RETURN 
            rec AS product,
            cat AS category,
            similar_customer_count AS score
        ORDER BY score DESC, rec.name
        LIMIT $top_k
        """

        results = await self.client.execute_read(
            query,
            {
                "customer_id": customer_id,
                "top_k": top_k,
                "min_shared_purchases": min_shared_purchases,
            },
        )

        return results

    async def get_category_recommendations(
        self,
        category_id: str,
        top_k: int = 10,
        min_purchases: int = 1,
    ) -> list[dict[str, Any]]:
        """Get popular products in a category.

        Args:
            category_id: Category ID (e.g., "section_16", "garment_1010", "type_253")
            top_k: Number of products to return
            min_purchases: Minimum purchases to include product

        Returns:
            List of products with purchase counts and category info
        """
        query = """
        MATCH (cat:Category {id: $category_id})<-[:IN_CATEGORY]-(p:Product)
        
        // Count purchases for each product
        OPTIONAL MATCH (c:Customer)-[pur:PURCHASED]->(p)
        WITH p, cat, count(pur) AS purchase_count, avg(pur.price) AS avg_price
        WHERE purchase_count >= $min_purchases
        
        // Get similar products for diversity
        OPTIONAL MATCH (p)-[sim:SIMILAR_TO]->(similar:Product)
        
        RETURN 
            p AS product,
            cat AS category,
            purchase_count,
            avg_price,
            collect(DISTINCT similar)[..3] AS similar_products
        ORDER BY purchase_count DESC, avg_price DESC
        LIMIT $top_k
        """

        results = await self.client.execute_read(
            query,
            {
                "category_id": category_id,
                "top_k": top_k,
                "min_purchases": min_purchases,
            },
        )

        return results

    async def get_trending_products(
        self,
        days: int = 7,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Get trending products based on recent purchases.

        Args:
            days: Number of days to look back
            top_k: Number of products to return

        Returns:
            List of trending products with recent purchase counts
        """
        query = """
        // Get recent purchases (within N days)
        MATCH (c:Customer)-[pur:PURCHASED]->(p:Product)
        WHERE pur.timestamp >= datetime() - duration({days: $days})
        
        // Count purchases per product
        WITH p, count(pur) AS recent_purchases, avg(pur.price) AS avg_price
        
        // Get category and similar products
        OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
        OPTIONAL MATCH (p)-[sim:SIMILAR_TO]->(similar:Product)
        
        RETURN 
            p AS product,
            cat AS category,
            recent_purchases,
            avg_price,
            collect(DISTINCT {product: similar, score: sim.score})[..3] AS similar_products
        ORDER BY recent_purchases DESC, avg_price DESC
        LIMIT $top_k
        """

        results = await self.client.execute_read(
            query,
            {"days": days, "top_k": top_k},
        )

        return results

    async def get_product_stats(self, product_id: int) -> dict[str, Any]:
        """Get statistics for a product.

        Args:
            product_id: Article ID of the product

        Returns:
            Dict with purchase count, unique customers, avg price, etc.
        """
        query = """
        MATCH (p:Product {id: $product_id})
        
        // Count purchases and customers
        OPTIONAL MATCH (c:Customer)-[pur:PURCHASED]->(p)
        WITH p, count(pur) AS total_purchases, count(DISTINCT c) AS unique_customers, avg(pur.price) AS avg_price
        
        // Get category
        OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
        
        // Count similar products
        OPTIONAL MATCH (p)-[:SIMILAR_TO]->(similar:Product)
        WITH p, cat, total_purchases, unique_customers, avg_price, count(similar) AS similar_count
        
        RETURN 
            p AS product,
            cat AS category,
            total_purchases,
            unique_customers,
            avg_price,
            similar_count
        """

        results = await self.client.execute_read(query, {"product_id": product_id})

        if not results:
            return {}

        return results[0]

    async def find_complementary_products(
        self,
        product_id: int,
        top_k: int = 5,
        min_co_purchases: int = 2,
    ) -> list[dict[str, Any]]:
        """Find products frequently bought together (complementary).

        This is different from similar products - it finds products that
        customers buy TOGETHER in their shopping baskets.

        Args:
            product_id: Article ID of the product
            top_k: Number of complementary products to return
            min_co_purchases: Minimum co-purchases to consider

        Returns:
            List of complementary products with co-purchase counts
        """
        query = """
        MATCH (p:Product {id: $product_id})<-[:PURCHASED]-(c:Customer)
        MATCH (c)-[:PURCHASED]->(comp:Product)
        WHERE comp.id <> p.id
        
        // Count co-purchases
        WITH p, comp, count(DISTINCT c) AS co_purchase_count
        WHERE co_purchase_count >= $min_co_purchases
        
        // Get category info
        OPTIONAL MATCH (comp)-[:IN_CATEGORY]->(cat:Category)
        OPTIONAL MATCH (p)-[:IN_CATEGORY]->(p_cat:Category)
        
        // Prefer different categories (cross-category recommendations)
        WITH comp, cat, p_cat, co_purchase_count,
             CASE WHEN cat.id <> p_cat.id THEN 1.5 ELSE 1.0 END AS cross_category_boost
        
        RETURN 
            comp AS product,
            cat AS category,
            co_purchase_count,
            (co_purchase_count * cross_category_boost) AS score
        ORDER BY score DESC, co_purchase_count DESC
        LIMIT $top_k
        """

        results = await self.client.execute_read(
            query,
            {
                "product_id": product_id,
                "top_k": top_k,
                "min_co_purchases": min_co_purchases,
            },
        )

        return results
