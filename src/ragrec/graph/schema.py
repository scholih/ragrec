"""Neo4j graph schema definitions."""

# Cypher schema for RagRec graph database

# ============================================================================
# CONSTRAINTS & INDEXES
# ============================================================================

CONSTRAINTS = [
    # Node uniqueness constraints
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT category_id IF NOT EXISTS FOR (cat:Category) REQUIRE cat.id IS UNIQUE",
    "CREATE CONSTRAINT persona_id IF NOT EXISTS FOR (per:Persona) REQUIRE per.id IS UNIQUE",
]

INDEXES = [
    # Frequently queried properties
    "CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name)",
    "CREATE INDEX category_name IF NOT EXISTS FOR (cat:Category) ON (cat.name)",
    "CREATE INDEX category_level IF NOT EXISTS FOR (cat:Category) ON (cat.level)",
]

# ============================================================================
# NODE CREATION QUERIES
# ============================================================================

CREATE_PRODUCT_NODES = """
UNWIND $products AS product
MERGE (p:Product {id: product.id})
SET p.name = product.name,
    p.product_type = product.product_type,
    p.colour_group = product.colour_group,
    p.department = product.department,
    p.section = product.section,
    p.garment_group = product.garment_group,
    p.index_group = product.index_group
"""

CREATE_CUSTOMER_NODES = """
UNWIND $customers AS customer
MERGE (c:Customer {id: customer.id})
SET c.age_bracket = customer.age_bracket,
    c.club_member_status = customer.club_member_status,
    c.fashion_news_frequency = customer.fashion_news_frequency
"""

CREATE_CATEGORY_NODES = """
UNWIND $categories AS category
MERGE (cat:Category {id: category.id})
SET cat.name = category.name,
    cat.level = category.level,
    cat.parent_id = category.parent_id
"""

CREATE_PERSONA_NODES = """
UNWIND $personas AS persona
MERGE (per:Persona {id: persona.id})
SET per.name = persona.name,
    per.description = persona.description,
    per.size = persona.size
"""

# ============================================================================
# RELATIONSHIP CREATION QUERIES
# ============================================================================

CREATE_PURCHASED_RELATIONSHIPS = """
UNWIND $purchases AS purchase
MATCH (c:Customer {id: purchase.customer_id})
MATCH (p:Product {id: purchase.product_id})
MERGE (c)-[r:PURCHASED]->(p)
SET r.timestamp = datetime(purchase.timestamp),
    r.price = purchase.price
"""

CREATE_IN_CATEGORY_RELATIONSHIPS = """
MATCH (p:Product)
MATCH (cat:Category)
WHERE cat.name = p.product_type AND cat.level = 'product_type'
MERGE (p)-[:IN_CATEGORY]->(cat)
"""

CREATE_PARENT_OF_RELATIONSHIPS = """
MATCH (child:Category)
WHERE child.parent_id IS NOT NULL
MATCH (parent:Category {id: child.parent_id})
MERGE (parent)-[:PARENT_OF]->(child)
"""

CREATE_SIMILAR_TO_RELATIONSHIPS = """
UNWIND $similarities AS sim
MATCH (p1:Product {id: sim.product_id})
MATCH (p2:Product {id: sim.similar_id})
MERGE (p1)-[r:SIMILAR_TO]->(p2)
SET r.score = sim.score,
    r.source = 'visual'
"""

CREATE_BELONGS_TO_RELATIONSHIPS = """
UNWIND $memberships AS membership
MATCH (c:Customer {id: membership.customer_id})
MATCH (per:Persona {id: membership.persona_id})
MERGE (c)-[:BELONGS_TO]->(per)
"""

# ============================================================================
# QUERY TEMPLATES
# ============================================================================

GET_PRODUCT_NEIGHBORHOOD = """
MATCH (p:Product {id: $product_id})
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (p)-[sim:SIMILAR_TO]->(similar:Product)
OPTIONAL MATCH (customer)-[pur:PURCHASED]->(p)
RETURN p,
       cat,
       collect(DISTINCT {product: similar, score: sim.score})[..5] AS similar_products,
       count(DISTINCT customer) AS purchase_count,
       avg(pur.price) AS avg_price
"""

GET_CUSTOMER_PURCHASES = """
MATCH (c:Customer {id: $customer_id})-[pur:PURCHASED]->(p:Product)
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
RETURN p, cat, pur
ORDER BY pur.timestamp DESC
LIMIT $limit
"""

GET_CATEGORY_HIERARCHY = """
MATCH path = (root:Category)-[:PARENT_OF*]->(leaf:Category)
WHERE root.level = 'section'
  AND leaf.level = 'product_type'
RETURN path
LIMIT $limit
"""

GET_PERSONA_CUSTOMERS = """
MATCH (per:Persona {id: $persona_id})<-[:BELONGS_TO]-(c:Customer)
RETURN c
LIMIT $limit
"""

# ============================================================================
# UTILITY QUERIES
# ============================================================================

CLEAR_ALL_DATA = """
MATCH (n)
DETACH DELETE n
"""

COUNT_NODES_BY_LABEL = """
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

COUNT_RELATIONSHIPS_BY_TYPE = """
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
