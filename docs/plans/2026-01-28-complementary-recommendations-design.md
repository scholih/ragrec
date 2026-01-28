# Complementary Recommendations: LLM-Enhanced Cross-Category System

**Design Document**
**Date:** 2026-01-28
**Status:** Approved for Implementation
**Author:** Collaborative design session

---

## Executive Summary

Extends RagRec beyond visual similarity to include **complementary recommendations** - items that "go well with" rather than "look similar to" the query item. Examples:
- Fashion + Fashion: trousers → shoes, belt, shirt
- Cross-category: fashion → cosmetics, accessories

**Key Innovation:** LLM-enhanced (Qwen) rule generation combined with transaction co-occurrence, exposed through LangChain natural language interface for tunable recommendation behavior.

**Design Principles:**
- **EU-Sovereign:** Self-hosted Qwen (already used for text embeddings), no external LLM APIs
- **Cost:** $0 - all local compute on M2 Pro
- **Tunable:** Users express intent ("bold combinations", "safe pairings") → system adjusts parameters
- **Hybrid:** Transaction data (high confidence) + LLM rules (broad coverage) + LLM product-specific (gap filling)

---

## Table of Contents

1. [Problem & Motivation](#1-problem--motivation)
2. [Data Sources for COMPLEMENTS Edges](#2-data-sources-for-complements-edges)
3. [LangChain Agent Architecture](#3-langchain-agent-architecture)
4. [LLM Rule Generation Pipeline](#4-llm-rule-generation-pipeline)
5. [Tunable Parameters & Experimentation](#5-tunable-parameters--experimentation)
6. [Integration with Visual Similarity](#6-integration-with-visual-similarity)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Technical Specifications](#8-technical-specifications)

---

## 1. Problem & Motivation

### Current State (Phase 1.3)
Visual similarity works excellently for **similar products**:
- Red dress → other red dresses
- Black ankle boots → similar black boots
- Query: "Find items that look like this"

### Gap: Complementary Recommendations
Fashion shopping requires **non-similar** recommendations:
- "What goes well with this dress?" → shoes, belt, accessories
- "Complete the outfit" → cross-category pairings
- "Show me bold combinations" vs "safe, classic pairings"

### Why This Matters
- **Basket size:** Complementary recommendations drive multi-item purchases
- **Discovery:** Help customers find items outside their query category
- **Differentiation:** Most e-commerce only does "similar items" or "frequently bought together"
- **Conversational:** Natural language interface lets users tune recommendation behavior

---

## 2. Data Sources for COMPLEMENTS Edges

Three complementary sources feed into Neo4j `(:Product)-[:COMPLEMENTS]->(:Product)` relationships:

### Source 1: Transaction Co-Occurrence (Free, High Signal)

**Logic:**
- Analyze 31.8M H&M transactions
- Find: customer bought items A + B within **7-day window**
- Rationale: Fashion purchases are sequential (dress Monday, shoes Wednesday, accessories Friday)

**Implementation:**
```python
# Polars query on transactions table
transactions = pl.read_parquet("data/hm/transactions.parquet")

co_occurrences = (
    transactions
    .sort("t_dat")
    .groupby("customer_id")
    .agg([
        pl.col("article_id").alias("items"),
        pl.col("t_dat").alias("dates")
    ])
    .explode(["items", "dates"])
    .join_asof(
        transactions.select(["customer_id", "article_id", "t_dat"]),
        by="customer_id",
        left_on="dates",
        right_on="t_dat",
        strategy="forward",
        tolerance=timedelta(days=7)
    )
    .filter(pl.col("article_id") != pl.col("article_id_right"))
    .groupby(["article_id", "article_id_right"])
    .agg(pl.count().alias("strength"))
)

# Store in Neo4j:
# (:Product {id: A})-[:COMPLEMENTS {strength: count, source: "transaction"}]->(:Product {id: B})
```

**Tunable Parameters:**
- `time_window`: 7 days (default), can vary 1-30 days
- `min_strength`: Minimum co-occurrences to create edge (default: 2, tunable 1-10)

### Source 2: LLM-Generated Category Rules (Free, Broad Coverage)

**Logic:**
- Use Qwen to generate category-level complementary relationships
- Example: "Trousers" → ["Shirts", "Shoes", "Belts", "Blazers"]
- Apply rules to all products in those categories

**Prompt Template:**
```python
prompt = f"""
You are a fashion stylist. What product categories naturally complement {category}?
Consider practical outfit building and style coherence.
List 3-5 categories that customers typically pair with {category}.
Return ONLY a JSON array of category names.

Example for "Dresses":
["Heels", "Clutches", "Jewelry", "Cardigans"]

Category: {category}
"""
```

**Storage:**
```cypher
// Category-level relationship
(:Category {name: "Trousers"})-[:COMPLEMENTS {source: "llm_category"}]->(:Category {name: "Shoes"})

// Propagated to products
MATCH (p:Product)-[:IN_CATEGORY]->(c1:Category)-[:COMPLEMENTS]->(c2:Category)<-[:IN_CATEGORY]-(comp:Product)
CREATE (p)-[:COMPLEMENTS {source: "llm_category", strength: 0.5}]->(comp)
```

### Source 3: LLM-Generated Product-Specific (Free, Precise)

**Logic:**
- For products with sparse transaction data (<3 COMPLEMENTS edges)
- Generate specific product-level recommendations
- Used for: new products, niche items, cross-category exploration

**Prompt Template:**
```python
prompt = f"""
Product: {product.name} ({product.category})
Description: {product.detail_desc}
Color: {product.colour_group_name}

Suggest 5 specific products that would complement this item.
Focus on:
- Different categories (not similar items)
- Practical outfit combinations
- Style coherence (consider color, formality, season)

Return JSON array:
[
  {{"category": "Shoes", "style": "ankle boots", "color_hint": "black or brown", "reason": "completes casual outfit"}},
  ...
]
"""
```

**When Generated:**
- Weekly batch: All products with <3 transaction-based complements
- On-demand: New products added to catalog
- Monthly refresh: Category rules (fashion trends evolve)

### Mixing Strategy

All three sources coexist in Neo4j with metadata:

```cypher
(:Product)-[:COMPLEMENTS {
  source: "transaction" | "llm_category" | "llm_product",
  strength: float,          // co-occurrence count OR llm confidence (0-1)
  generated_at: timestamp,
  reasoning: string         // optional LLM explanation
}]->(:Product)
```

**Query-Time Weighting:**
```python
# User query with tunable weights
params = {
    "transaction_weight": 0.7,    # Trust real behavior
    "llm_rule_weight": 0.3,       // Add AI creativity
    "min_strength": 2             // Filter threshold
}

# Cypher query
MATCH (p:Product {id: $product_id})-[c:COMPLEMENTS]->(comp:Product)
WHERE c.strength >= $min_strength
RETURN comp,
       CASE c.source
         WHEN 'transaction' THEN c.strength * $transaction_weight
         WHEN 'llm_category' THEN 0.5 * $llm_rule_weight
         WHEN 'llm_product' THEN c.strength * $llm_rule_weight
       END as score
ORDER BY score DESC
LIMIT $top_k
```

---

## 3. LangChain Agent Architecture

### Natural Language → Graph Query

Users express recommendation intent verbally → LangChain interprets → System adjusts parameters

**Flow:**
```
User: "Show me bold, unexpected outfit combinations"
  ↓
LangChain Agent interprets intent
  ↓
Parameters: {
  similarity_weight: 0.2,       # Low - we want different
  cross_category_boost: 0.8,    # High - mix categories
  transaction_weight: 0.3,      # Low - ignore safe pairs
  llm_rule_weight: 0.7,         # High - use AI creativity
  novelty_penalty: 0.0,         # No penalty for unusual
  min_complement_strength: 2    # Low threshold
}
  ↓
Execute Neo4j query with weighted scoring
  ↓
Return ranked complementary products
```

### LangChain Tools

```python
from langchain.tools import tool
from langchain.agents import create_react_agent

@tool
async def search_similar(product_id: int, top_k: int = 10) -> list:
    """Find visually similar products (existing Phase 1.3)"""
    # Uses VisualRecommender
    pass

@tool
async def search_complements(
    product_id: int,
    user_intent: str,
    top_k: int = 10
) -> list:
    """Find complementary products with natural language intent"""
    params = await interpret_style_intent(user_intent)
    return await graph_client.find_complements(product_id, params, top_k)

@tool
async def interpret_style_intent(user_query: str) -> dict:
    """
    Translate natural language to recommendation parameters.

    Examples:
    - "conservative pairings" → {similarity_weight: 0.9, cross_category_boost: 0.1}
    - "surprise me" → {novelty_penalty: 0.0, llm_rule_weight: 0.8}
    - "professional but modern" → blend profile
    """

    prompt = f"""
    User wants: "{user_query}"

    Map this to recommendation parameters (0.0-1.0 scale):

    - similarity_weight: How similar to query item? (1.0 = very similar, 0.0 = very different)
    - cross_category_boost: Mix different categories? (1.0 = yes cross freely, 0.0 = stay in category)
    - transaction_weight: Trust customer behavior? (1.0 = only proven pairs, 0.0 = ignore history)
    - llm_rule_weight: Trust AI suggestions? (1.0 = creative AI pairings, 0.0 = ignore AI)
    - novelty_penalty: Penalize popular items? (1.0 = only show unique, 0.0 = allow popular)

    Intent keywords mapping:
    - "bold", "unexpected", "surprise" → low similarity, high cross-category, low transaction
    - "safe", "classic", "conservative" → high similarity, low cross-category, high transaction
    - "professional" → moderate all, slight transaction bias
    - "creative", "artistic" → high llm_rule, low transaction

    Return JSON with values 0.0-1.0 for each parameter.
    Format: {{"similarity_weight": 0.5, "cross_category_boost": 0.7, ...}}
    """

    return await qwen_client.generate(prompt, format="json")

@tool
async def explain_recommendation(product_id: int, recommended_id: int) -> str:
    """Explain why a product was recommended"""
    # Query Neo4j for relationship path
    # Return natural language explanation
    pass
```

### Agent Configuration

```python
from langchain_community.llms import Ollama

# Use local Qwen (already running for embeddings)
llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434")

tools = [
    search_similar,
    search_complements,
    interpret_style_intent,
    explain_recommendation
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=recommendation_agent_prompt
)
```

---

## 4. LLM Rule Generation Pipeline

### Batch Generation Process

**Phase 1: Category-Level Rules (One-Time + Monthly Refresh)**

```python
async def generate_category_complements():
    """
    Generate COMPLEMENTS relationships between product categories.
    Cost: ~$0 (local Qwen), Time: ~5-10 minutes for 50 categories
    """

    # Get unique categories from H&M data
    categories = await db.fetch("""
        SELECT DISTINCT product_type_name
        FROM products
        WHERE product_type_name IS NOT NULL
    """)

    for category in categories:
        prompt = f"""
        You are a fashion stylist with expertise in outfit coordination.

        Category: {category}

        What product categories naturally complement {category} in a complete outfit?
        Consider:
        - Practical outfit building (what do customers actually wear together?)
        - Style coherence (formality levels, occasions)
        - Cross-category opportunities (not just similar items)

        List 3-5 complementary categories.
        Return ONLY a JSON array of category names.

        Example for "Dresses":
        ["Heels", "Clutches", "Jewelry", "Cardigans", "Belts"]

        Your answer for "{category}":
        """

        response = await qwen_client.generate(prompt, format="json")
        complements = json.loads(response)

        # Store in Neo4j
        await graph_client.execute("""
            MATCH (c1:Category {name: $category})
            UNWIND $complements as comp_name
            MATCH (c2:Category {name: comp_name})
            MERGE (c1)-[:COMPLEMENTS {
                source: 'llm_category',
                generated_at: datetime(),
                confidence: 0.8
            }]->(c2)
        """, category=category, complements=complements)

    # Propagate to products
    await propagate_category_rules_to_products()
```

**Phase 2: Product-Specific Gap Filling (Weekly Batch)**

```python
async def generate_product_complements_for_sparse_items():
    """
    Generate product-level complements for items with insufficient transaction data.
    Runs weekly for new products or items with <3 COMPLEMENTS edges.
    """

    sparse_products = await db.fetch("""
        SELECT p.article_id, p.prod_name, p.detail_desc,
               p.product_type_name, p.colour_group_name,
               COUNT(c.*) as complement_count
        FROM products p
        LEFT JOIN product_embeddings pe ON p.article_id = pe.article_id
        LEFT JOIN complements c ON p.article_id = c.from_product_id
        WHERE pe.embedding IS NOT NULL  -- Only products with embeddings
        GROUP BY p.article_id, p.prod_name, p.detail_desc,
                 p.product_type_name, p.colour_group_name
        HAVING COUNT(c.*) < 3
    """)

    for product in sparse_products:
        prompt = f"""
        You are a fashion stylist creating outfit recommendations.

        Product Details:
        - Name: {product['prod_name']}
        - Category: {product['product_type_name']}
        - Color: {product['colour_group_name']}
        - Description: {product['detail_desc']}

        Suggest 5 specific products that would complement this item in an outfit.
        Requirements:
        - Focus on DIFFERENT categories (not similar items in same category)
        - Consider practical outfit combinations for real shopping
        - Match style coherence (color harmony, formality, occasion)
        - Suggest cross-category items (e.g., fashion + accessories)

        Return JSON array with specific recommendations:
        [
          {{
            "category": "Shoes",
            "style": "black ankle boots",
            "color_hint": "black or dark brown",
            "reasoning": "Completes casual-chic outfit, matches neutral tones"
          }},
          ...
        ]

        Return ONLY the JSON array, no other text.
        """

        response = await qwen_client.generate(prompt, format="json")
        suggestions = json.loads(response)

        # Match suggestions to actual products in catalog
        for suggestion in suggestions:
            matching_products = await find_matching_products(
                category=suggestion['category'],
                style_keywords=suggestion['style'],
                color_hint=suggestion['color_hint']
            )

            for match in matching_products[:2]:  # Top 2 matches per suggestion
                await graph_client.execute("""
                    MERGE (p1:Product {id: $from_id})
                    MERGE (p2:Product {id: $to_id})
                    MERGE (p1)-[:COMPLEMENTS {
                        source: 'llm_product',
                        strength: 0.7,
                        generated_at: datetime(),
                        reasoning: $reasoning
                    }]->(p2)
                """,
                from_id=product['article_id'],
                to_id=match['article_id'],
                reasoning=suggestion['reasoning'])
```

**Regeneration Schedule:**

| Task | Frequency | Trigger | Cost |
|------|-----------|---------|------|
| Category rules | Monthly | Cron: 1st of month 2 AM | $0, ~10 min |
| Product-specific | Weekly | Cron: Sunday 3 AM | $0, ~30-60 min |
| On-demand | As needed | New product webhook | $0, ~1-2 sec/product |

---

## 5. Tunable Parameters & Experimentation

### Recommendation Profiles

Pre-defined configurations for common use cases:

```yaml
# config/recommendation_profiles.yaml

safe_classic:
  description: "Conservative, proven pairings from customer behavior"
  similarity_weight: 0.8
  cross_category_boost: 0.2
  transaction_weight: 0.7
  llm_rule_weight: 0.3
  novelty_penalty: 0.8
  min_complement_strength: 5

bold_experimental:
  description: "Unexpected, creative combinations"
  similarity_weight: 0.2
  cross_category_boost: 0.8
  transaction_weight: 0.3
  llm_rule_weight: 0.7
  novelty_penalty: 0.0
  min_complement_strength: 2

professional_modern:
  description: "Balanced business-appropriate with contemporary edge"
  similarity_weight: 0.6
  cross_category_boost: 0.4
  transaction_weight: 0.6
  llm_rule_weight: 0.4
  novelty_penalty: 0.5
  min_complement_strength: 3

trend_forward:
  description: "Focus on new arrivals and unique items"
  similarity_weight: 0.3
  cross_category_boost: 0.7
  transaction_weight: 0.2
  llm_rule_weight: 0.8
  novelty_penalty: 0.2
  min_complement_strength: 2
  additional_filters:
    - "created_at > now() - interval '30 days'"  # Recent products
```

### Dynamic Parameter Interpretation

LangChain translates free-form text → profile blend:

```python
# Examples of natural language → parameter mapping

"Show me safe, classic pairings"
→ Load profile: safe_classic

"I want to stand out at a party"
→ Load profile: bold_experimental
→ Override: novelty_penalty = 0.0

"Professional but not boring"
→ Blend: 60% professional_modern + 40% trend_forward

"Surprise me, but nothing too crazy"
→ Load profile: bold_experimental
→ Override: min_complement_strength = 4 (higher threshold)

"Complete this outfit for a wedding"
→ Load profile: professional_modern
→ Override: cross_category_boost = 0.9 (need accessories)
→ Add filter: formality_level = "formal"
```

### A/B Testing Framework

```python
async def compare_recommendation_strategies(product_id: int):
    """
    Run multiple strategies in parallel for experimentation.
    Returns side-by-side comparison for evaluation.
    """

    strategies = {
        "safe": "safe_classic",
        "bold": "bold_experimental",
        "balanced": "professional_modern"
    }

    results = {}

    for name, profile in strategies.items():
        params = load_profile(profile)
        results[name] = await recommend_complements(
            product_id=product_id,
            params=params,
            top_k=10
        )

    return {
        "query_product": await get_product(product_id),
        "strategies": results,
        "evaluation_metrics": {
            name: calculate_diversity_score(recs)
            for name, recs in results.items()
        }
    }
```

### Evaluation Metrics

```python
def calculate_diversity_score(recommendations: list) -> dict:
    """
    Measure recommendation quality dimensions.
    """

    categories = [r['category'] for r in recommendations]

    return {
        "category_diversity": len(set(categories)) / len(categories),
        "cross_category_ratio": sum(1 for c in categories if c != query_category) / len(categories),
        "avg_novelty": mean([r['popularity_percentile'] for r in recommendations]),
        "avg_confidence": mean([r['score'] for r in recommendations])
    }
```

---

## 6. Integration with Visual Similarity

### Unified Recommendation Engine

```python
# src/ragrec/recommender/engine.py

class RecommendationEngine:
    """
    Unified interface for all recommendation types:
    - Visual similarity (Phase 1.3)
    - Complementary (Phase 2.1)
    - Mixed/Fusion (Phase 4)
    """

    def __init__(self):
        self.visual_recommender = VisualRecommender()  # Existing
        self.graph_client = Neo4jClient()              # New
        self.langchain_agent = ComplementsAgent()      # New

    async def recommend(
        self,
        query_product_id: int,
        mode: Literal["similar", "complements", "mixed"] = "similar",
        user_intent: str | None = None,
        top_k: int = 10,
        filters: dict | None = None
    ) -> pl.DataFrame:
        """
        Universal recommendation method.

        Args:
            query_product_id: Reference product
            mode: Recommendation type
                - "similar": Visual similarity (look-alike products)
                - "complements": What goes with this (outfit completion)
                - "mixed": Interleaved similar + complements
            user_intent: Natural language style description
                - "bold combinations"
                - "safe, classic pairings"
                - "professional but modern"
            top_k: Number of results
            filters: Optional filters (category, price, etc.)

        Returns:
            Polars DataFrame with recommendations
        """

        if mode == "similar":
            # Existing visual similarity (Phase 1.3)
            return await self.visual_recommender.find_similar_by_id(
                product_id=query_product_id,
                top_k=top_k,
                category_filter=filters.get("category") if filters else None
            )

        elif mode == "complements":
            # New complementary recommendations

            # Interpret user intent with LangChain
            if user_intent:
                params = await self.langchain_agent.interpret_style_intent(user_intent)
            else:
                params = load_profile("professional_modern")  # Default

            # Execute graph query
            return await self.graph_client.find_complements(
                product_id=query_product_id,
                params=params,
                top_k=top_k,
                filters=filters
            )

        elif mode == "mixed":
            # Fusion: interleave similar + complements

            similar_results = await self.recommend(
                query_product_id,
                mode="similar",
                top_k=top_k // 2,
                filters=filters
            )

            complement_results = await self.recommend(
                query_product_id,
                mode="complements",
                user_intent=user_intent,
                top_k=top_k // 2,
                filters=filters
            )

            # Interleave: similar[0], complement[0], similar[1], complement[1], ...
            return self._interleave(similar_results, complement_results)

    def _interleave(self, df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
        """Interleave two DataFrames row by row."""
        combined = []
        for i in range(max(len(df1), len(df2))):
            if i < len(df1):
                combined.append(df1[i])
            if i < len(df2):
                combined.append(df2[i])
        return pl.DataFrame(combined)
```

### Enhanced API Endpoints

```python
# src/ragrec/api/main.py

@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend_products(
    product_id: int = Form(...),
    mode: Literal["similar", "complements", "mixed"] = Form("similar"),
    user_intent: str | None = Form(None),
    top_k: int = Form(10),
    category_filter: str | None = Form(None)
) -> RecommendationResponse:
    """
    Universal recommendation endpoint.

    Examples:
    - Visual similarity: mode=similar
    - Outfit completion: mode=complements, user_intent="complete this outfit"
    - Bold pairings: mode=complements, user_intent="show me unexpected combinations"
    - Mixed view: mode=mixed
    """

    engine = RecommendationEngine()

    results = await engine.recommend(
        query_product_id=product_id,
        mode=mode,
        user_intent=user_intent,
        top_k=top_k,
        filters={"category": category_filter} if category_filter else None
    )

    return RecommendationResponse(
        mode=mode,
        user_intent=user_intent,
        results=[ProductScore(**row) for row in results.iter_rows(named=True)],
        count=len(results)
    )

# Legacy endpoints (backward compatible)
@app.post("/api/v1/similar", response_model=SimilarityResponse)
async def find_similar_products(...):
    """Existing Phase 1.3 endpoint - redirects to /recommend with mode=similar"""
    # Backward compatible wrapper
    pass

@app.post("/api/v1/complements", response_model=ComplementsResponse)
async def find_complementary_products(
    product_id: int = Form(...),
    user_intent: str = Form("professional but modern"),
    top_k: int = Form(10)
):
    """New complementary-only endpoint"""
    # Wrapper around /recommend with mode=complements
    pass
```

### Enhanced CLI Commands

```bash
# Existing (Phase 1.3)
ragrec search <image> --top-k 10 --category "Dress"
ragrec similar <product-id> --top-k 10

# New (Phase 2.1)
ragrec complements <product-id> \
  --intent "show me bold, unexpected pairings" \
  --top-k 10

ragrec complements 822115001 \
  --intent "complete this outfit for a wedding" \
  --top-k 5

ragrec recommend <product-id> \
  --mode mixed \
  --intent "professional but modern" \
  --top-k 20

# A/B testing
ragrec compare-strategies <product-id> \
  --strategies safe,bold,balanced \
  --top-k 10
```

---

## 7. Implementation Roadmap

### Extended Phase 2 (Neo4j + Complementary System)

**Phase 2.0: Neo4j Foundation** (Original, ~3-4 days)
- [x] Phase 1.3 complete (prerequisite)
- [ ] Neo4j client with async driver
- [ ] Schema: Customer, Product, Category, Persona nodes
- [ ] Relationships: PURCHASED, IN_CATEGORY, PARENT_OF
- [ ] Bulk import from PostgreSQL (1000 products, 500 customers, 5000 transactions)
- [ ] Create SIMILAR_TO edges from existing visual similarity (top-5 per product)
- [ ] Indexes and constraints

**Phase 2.1: Complementary Recommendations** (~4-5 days)
- [ ] Transaction co-occurrence analysis
  - [ ] Polars ETL: 7-day window co-purchase detection
  - [ ] Create COMPLEMENTS edges (source: "transaction")
  - [ ] Tunable threshold parameter (default: 2)
- [ ] LLM category-level rules
  - [ ] Qwen client wrapper (reuse existing text embedding client)
  - [ ] Generate category→category complementary relationships
  - [ ] Store in Neo4j, propagate to products
- [ ] LLM product-specific gap filling
  - [ ] Identify sparse products (<3 complements)
  - [ ] Generate product-specific recommendations
  - [ ] Match suggestions to catalog items
- [ ] Graph queries for weighted complementary search
  - [ ] Cypher templates with parameter substitution
  - [ ] Scoring: transaction_weight + llm_rule_weight blend
- [ ] Integration tests

**Phase 2.2: LangChain Agent** (~3-4 days)
- [ ] Intent interpretation tool
  - [ ] Natural language → parameter dict
  - [ ] Profile loading (safe_classic, bold_experimental, etc.)
- [ ] LangChain agent setup
  - [ ] Tools: search_similar, search_complements, explain
  - [ ] ReAct agent with Qwen backend
- [ ] RecommendationEngine class
  - [ ] Unified interface: similar/complements/mixed modes
  - [ ] Interleaving logic for mixed mode
- [ ] API endpoints
  - [ ] POST /api/v1/recommend (universal)
  - [ ] POST /api/v1/complements (complementary-only)
  - [ ] Backward compatible /api/v1/similar
- [ ] CLI commands
  - [ ] `ragrec complements`
  - [ ] `ragrec recommend --mode mixed`
  - [ ] `ragrec compare-strategies`
- [ ] Integration tests with LangChain

**Phase 2.3: Evaluation & Tuning** (~2-3 days)
- [ ] A/B testing framework
  - [ ] Compare strategies side-by-side
  - [ ] Diversity metrics (category spread, cross-category ratio)
- [ ] Manual evaluation UI (Streamlit)
  - [ ] Show product + 3 strategy results
  - [ ] Thumbs up/down feedback
  - [ ] Adjust parameters interactively
- [ ] Documentation
  - [ ] Profile configuration guide
  - [ ] LangChain prompt engineering tips

**Total: ~12-16 days**

---

## 8. Technical Specifications

### File Structure

```
src/ragrec/
├── recommender/
│   ├── visual.py              # Existing (Phase 1.3)
│   ├── complements.py         # NEW: Complementary search logic
│   ├── engine.py              # NEW: Unified RecommendationEngine
│   └── fusion.py              # Future: Advanced fusion strategies
│
├── graph/
│   ├── __init__.py
│   ├── client.py              # Neo4j async client
│   ├── schema.py              # Cypher DDL
│   ├── queries.py             # Query templates
│   └── complements_builder.py # NEW: Build COMPLEMENTS edges
│
├── agents/                    # NEW: LangChain integration
│   ├── __init__.py
│   ├── intent_interpreter.py # Natural language → params
│   ├── tools.py              # LangChain tool definitions
│   └── profiles.py           # Load YAML profiles
│
├── llm/                       # NEW: LLM utilities
│   ├── __init__.py
│   ├── qwen_client.py        # Wrapper for Qwen (reuse existing)
│   └── rule_generator.py     # Generate complementary rules
│
└── api/
    └── routers/
        └── recommendations.py # NEW: Unified /recommend endpoint

config/
└── recommendation_profiles.yaml  # NEW: Profile definitions

tests/
├── unit/
│   ├── test_complements_builder.py
│   ├── test_intent_interpreter.py
│   └── test_recommendation_engine.py
│
├── integration/
│   ├── test_complements_recommendations.py
│   ├── test_langchain_agent.py
│   └── test_mixed_mode.py
│
└── evaluation/
    ├── test_strategy_comparison.py
    └── manual_eval_ui.py       # Streamlit app
```

### Dependencies (Additional)

```toml
[project.dependencies]
# Existing dependencies from Phase 1.3...

# Graph
"neo4j>=5.20.0",

# LangChain
"langchain>=0.1.0",
"langchain-community>=0.0.20",

# Local LLM (Ollama/Qwen already installed)
# No new dependencies - reuse existing setup
```

### Neo4j Schema (Extended)

```cypher
// Nodes (from original design + complementary enhancements)
(:Customer {id, age_bracket, persona_id, behavior_embedding})
(:Product {id, name, category, image_embedding, text_embedding})
(:Category {id, name, level})
(:Persona {id, name, description, centroid_embedding})

// Relationships (extended)
(:Customer)-[:PURCHASED {timestamp, price, channel}]->(:Product)
(:Customer)-[:BELONGS_TO]->(:Persona)
(:Product)-[:IN_CATEGORY]->(:Category)
(:Category)-[:PARENT_OF]->(:Category)

// Visual similarity (Phase 1.3)
(:Product)-[:SIMILAR_TO {score: float, source: "visual"}]->(:Product)

// NEW: Complementary relationships (Phase 2.1)
(:Product)-[:COMPLEMENTS {
  source: "transaction" | "llm_category" | "llm_product",
  strength: float,
  generated_at: datetime,
  reasoning: string
}]->(:Product)

(:Category)-[:COMPLEMENTS {
  source: "llm_category",
  generated_at: datetime
}]->(:Category)

// Indexes
CREATE INDEX product_id FOR (p:Product) ON (p.id);
CREATE INDEX category_name FOR (c:Category) ON (c.name);
CREATE INDEX complement_source FOR ()-[r:COMPLEMENTS]-() ON (r.source);
CREATE INDEX complement_strength FOR ()-[r:COMPLEMENTS]-() ON (r.strength);
```

### Example Cypher Queries

**Find Complements with Weighted Scoring:**
```cypher
MATCH (p:Product {id: $product_id})-[c:COMPLEMENTS]->(comp:Product)
WHERE c.strength >= $min_strength
RETURN
  comp.id as product_id,
  comp.name as name,
  comp.category as category,
  c.source as source,
  CASE c.source
    WHEN 'transaction' THEN c.strength * $transaction_weight
    WHEN 'llm_category' THEN 0.5 * $llm_rule_weight
    WHEN 'llm_product' THEN c.strength * $llm_rule_weight
  END as score,
  c.reasoning as explanation
ORDER BY score DESC
LIMIT $top_k
```

**Explain Recommendation Path:**
```cypher
MATCH path = (query:Product {id: $query_id})-[c:COMPLEMENTS]->(rec:Product {id: $rec_id})
RETURN
  query.name as query_product,
  rec.name as recommended_product,
  c.source as relationship_source,
  c.strength as confidence,
  c.reasoning as explanation,
  CASE c.source
    WHEN 'transaction' THEN 'Customers frequently bought these together'
    WHEN 'llm_category' THEN 'Fashion stylist suggests these categories pair well'
    WHEN 'llm_product' THEN c.reasoning
  END as user_explanation
```

### Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Complementary search (warm cache) | <100ms | Neo4j graph query |
| LLM intent interpretation | <2s | Local Qwen inference |
| LLM rule generation (batch) | ~10min | 50 categories, monthly |
| Product-specific generation | ~30-60min | Weekly batch, sparse items |
| Mixed mode (similar + complements) | <300ms | Parallel queries |

### Cost Analysis

| Component | Cost | Frequency |
|-----------|------|-----------|
| Transaction co-occurrence | $0 | One-time + weekly incremental |
| LLM category rules | $0 (local Qwen) | Monthly |
| LLM product-specific | $0 (local Qwen) | Weekly |
| Runtime queries | $0 | Per request |
| **Total** | **$0** | EU-sovereign, self-hosted |

---

## Appendix: Example Usage

### API Examples

**1. Visual Similarity (Existing)**
```bash
curl -X POST "http://localhost:9010/api/v1/similar" \
  -F "file=@dress.jpg" \
  -F "top_k=10"
```

**2. Complementary with Intent**
```bash
curl -X POST "http://localhost:9010/api/v1/recommend" \
  -F "product_id=822115001" \
  -F "mode=complements" \
  -F "user_intent=show me bold, unexpected pairings" \
  -F "top_k=10"
```

**3. Mixed Mode**
```bash
curl -X POST "http://localhost:9010/api/v1/recommend" \
  -F "product_id=822115001" \
  -F "mode=mixed" \
  -F "user_intent=professional but modern" \
  -F "top_k=20"
```

### CLI Examples

```bash
# Find complementary products with natural language
ragrec complements 822115001 \
  --intent "complete this outfit for a summer wedding" \
  --top-k 5

# Compare strategies
ragrec compare-strategies 822115001 \
  --strategies safe,bold,professional \
  --top-k 10

# Mixed recommendations
ragrec recommend 822115001 \
  --mode mixed \
  --intent "surprise me, but nothing too crazy" \
  --top-k 20
```

### Python SDK Example

```python
from ragrec.recommender import RecommendationEngine

engine = RecommendationEngine()

# Get complementary recommendations
results = await engine.recommend(
    query_product_id=822115001,
    mode="complements",
    user_intent="bold, artistic combinations",
    top_k=10
)

print(results)
# Output: Polars DataFrame with product_id, name, category, score, explanation
```

---

## References

- **Original Design:** `docs/plans/2025-01-28-ragrec-design.md`
- **Phase 1.3 Implementation:** Visual Similarity Search (completed)
- **LangChain Documentation:** https://python.langchain.com/docs/
- **Neo4j Cypher Manual:** https://neo4j.com/docs/cypher-manual/
- **Market Basket Analysis:** Association rule mining for co-occurrence patterns
