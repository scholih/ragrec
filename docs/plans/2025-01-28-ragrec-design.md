# RagRec: Multi-Modal E-Retail Recommendation System

**Design Document**
**Date:** 2025-01-28
**Status:** Approved for Implementation

---

## Executive Summary

RagRec is an enterprise-grade, GDPR-compliant recommendation system for e-retail, combining:
- **Visual similarity** via SigLIP embeddings
- **Customer personas** via hybrid clustering (embeddings + graph communities)
- **Knowledge graph reasoning** via Neo4j GraphRAG

The system is designed for EU-sovereign deployment (open source only, self-hosted) and optimized for Mac M2 Pro ARM-native development.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Model](#2-data-model)
3. [Embedding & Vector Store](#3-embedding--vector-store)
4. [Persona Modeling & GraphRAG](#4-persona-modeling--graphrag)
5. [Project Structure](#5-project-structure)
6. [Dependencies](#6-dependencies)
7. [CLI & API Design](#7-cli--api-design)
8. [Streamlit UI](#8-streamlit-ui)
9. [n8n Workflow Automation](#9-n8n-workflow-automation)
10. [Setup & Development](#10-setup--development)
11. [Implementation Phases](#11-implementation-phases)
12. [Experimental Ranker Evaluation](#12-experimental-ranker-evaluation)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RagRec Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│  Clients                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │ FastAPI  │  │Streamlit │  │   CLI    │                          │
│  │  REST    │  │  Demo UI │  │  Batch   │                          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                          │
│       └─────────────┼─────────────┘                                │
│                     ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Recommendation Engine                     │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐               │   │
│  │  │  Visual   │  │  Persona  │  │  GraphRAG │               │   │
│  │  │ Similarity│  │  Matching │  │  Reasoning│               │   │
│  │  │ (SigLIP)  │  │ (Hybrid)  │  │  (Neo4j)  │               │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘               │   │
│  │        └──────────────┼──────────────┘                      │   │
│  │                       ▼                                      │   │
│  │              Fusion & Ranking Layer                          │   │
│  │         (RRF / Weighted Merge + Cross-Encoder)               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│       ┌──────────────────────┼──────────────────────┐              │
│       ▼                      ▼                      ▼              │
│  ┌─────────┐          ┌───────────┐          ┌──────────┐         │
│  │pgvector │          │  Qdrant   │          │  Neo4j   │         │
│  │(primary)│          │(benchmark)│          │  Graph   │         │
│  └─────────┘          └───────────┘          └──────────┘         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    n8n Workflow Engine                       │   │
│  │  (ETL Orchestration, Webhooks, Scheduled Jobs, Alerts)       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **EU-Sovereign** | All open source, self-hosted, no US Big Tech dependencies |
| **GDPR-Compliant** | Pseudonymized IDs, behavioral patterns only, no PII |
| **ARM-Native** | Homebrew install, M2 Pro optimized, no Docker required |
| **Rust-Ready** | Abstract interfaces for future performance optimization |
| **Polars-First** | No Pandas - faster data processing throughout |
| **Benchmark-Driven** | pgvector vs Qdrant comparison built-in |

---

## 2. Data Model

### H&M Dataset Mapping

```
H&M Dataset                          RagRec Data Model
─────────────                        ─────────────────
articles.csv (106K)        →         products table + Neo4j nodes
  - article_id                         - id, name, description
  - prod_name, detail_desc             - category_id, subcategory
  - product_type, colour               - attributes (JSONB)
  - section_name, garment_group        - image_embedding (vector 768)
  - index_group_name                   - text_embedding (vector 768)

customers.csv (1.37M)      →         customers table + Neo4j nodes
  - customer_id                        - id (pseudonymized)
  - age                                - age_bracket (enum)
  - postal_code                        - behavior_embedding (vector 256)
  - club_member_status                 - persona_id (FK)
  - FN, Active                         - segment_tags (array)

transactions.csv (31.8M)   →         interactions table + Neo4j edges
  - t_dat, customer_id                 - timestamp, customer_id
  - article_id, price                  - product_id, price
  - sales_channel_id                   - channel, session_id

images/ (106K JPGs)        →         object storage + embeddings
  - {article_id}.jpg                   - products.image_embedding
```

### Neo4j Graph Schema

```cypher
// Nodes
(:Customer {id, age_bracket, persona_id, behavior_embedding})
(:Product {id, name, image_embedding, text_embedding})
(:Category {id, name, level})  // hierarchical
(:Persona {id, name, description, centroid_embedding})

// Relationships
(:Customer)-[:PURCHASED {timestamp, price, channel}]->(:Product)
(:Customer)-[:BELONGS_TO]->(:Persona)
(:Product)-[:IN_CATEGORY]->(:Category)
(:Product)-[:SIMILAR_TO {score}]->(:Product)  // visual similarity
(:Category)-[:PARENT_OF]->(:Category)
```

### Sample Data Strategy

- **Committed to repo**: `data/sample/` with 1000 products, 500 customers, 5000 transactions
- **Gitignored**: `data/hm/` (full dataset), `data/embeddings/` (generated)

---

## 3. Embedding & Vector Store

### Embedding Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Embedding Generation                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Product Images          Product Text           Customer History│
│       │                       │                       │         │
│       ▼                       ▼                       ▼         │
│  ┌─────────┐            ┌─────────┐            ┌──────────┐    │
│  │ SigLIP  │            │ SigLIP  │            │ Sequence │    │
│  │ Vision  │            │  Text   │            │ Encoder  │    │
│  │ Encoder │            │ Encoder │            │(custom)  │    │
│  └────┬────┘            └────┬────┘            └────┬─────┘    │
│       │ 768-dim              │ 768-dim              │ 256-dim  │
│       ▼                       ▼                       ▼         │
│  ┌─────────┐            ┌─────────┐            ┌──────────┐    │
│  │   PCA   │ (optional) │   PCA   │ (optional) │  Store   │    │
│  │ 768→128 │            │ 768→128 │            │  as-is   │    │
│  └────┬────┘            └────┬────┘            └────┬─────┘    │
│       └──────────┬───────────┘                      │          │
│                  ▼                                   ▼          │
│         ┌──────────────┐                   ┌──────────────┐    │
│         │ Vector Store │                   │ Vector Store │    │
│         │  (products)  │                   │ (customers)  │    │
│         └──────────────┘                   └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Model Choice: SigLIP

- **Model**: `google/siglip-base-patch16-256-multilingual`
- **Why**: Proven at Mercari (+50% CTR, +14% conversion), works with smaller batches than CLIP
- **Output**: 768-dimensional embeddings (optionally compressed to 128-dim via PCA)

### Vector Store Abstraction

```python
class VectorStore(ABC):
    """Abstract interface - swap pgvector/Qdrant without code changes"""

    @abstractmethod
    async def upsert(self, ids: list[str], embeddings: list[list[float]],
                     metadata: dict) -> None: ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int = 10,
                     filters: dict | None = None) -> pl.DataFrame: ...

    @abstractmethod
    async def hybrid_search(self, query_embedding: list[float],
                           query_text: str, top_k: int = 10) -> pl.DataFrame: ...
```

### Benchmark Configuration

Both stores tested with identical HNSW parameters:
- `ef_construction`: 128
- `m` (connections): 16
- Distance: Cosine

---

## 4. Persona Modeling & GraphRAG

### Hybrid Persona Discovery

```
Step 1: Behavioral Embeddings
  Customer Purchase History → Sequence Encoding
  [prod_1, prod_2, ...] → Transformer → 256-dim vector

Step 2: Graph Community Detection
  Neo4j Louvain on Customer-Product bipartite graph

Step 3: Embedding Clustering
  HDBSCAN on behavior embeddings

Step 4: Persona Fusion
  Persona = Graph Community ∩ Embedding Cluster
  - "Young Trend Seekers": community_3 ∩ cluster_7
  - "Premium Classics": community_1 ∩ cluster_2
```

### GraphRAG Query Flow

```python
async def graph_enhanced_recommend(customer_id: str, top_k: int = 10):
    # 1. Get customer's persona and graph context
    context = await neo4j.run("""
        MATCH (c:Customer {id: $cid})-[:BELONGS_TO]->(p:Persona)
        MATCH (c)-[:PURCHASED]->(bought:Product)
        MATCH (bought)-[:IN_CATEGORY]->(cat:Category)
        MATCH (bought)-[:SIMILAR_TO]->(similar:Product)
        WHERE NOT (c)-[:PURCHASED]->(similar)
        RETURN p, collect(DISTINCT cat) as categories,
               collect(DISTINCT similar)[..20] as candidates
    """, cid=customer_id)

    # 2. Visual similarity (if query image provided)
    # 3. Persona-weighted fusion
    # 4. Return top-k
```

---

## 5. Project Structure

```
ragrec/
├── pyproject.toml
├── CLAUDE.md
├── MANIFESTO.md
├── README.md
│
├── src/
│   └── ragrec/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── embeddings/
│       │   ├── base.py         # Abstract interface
│       │   ├── siglip.py       # SigLIP encoder
│       │   └── sequence.py     # Customer behavior encoder
│       │
│       ├── vectorstore/
│       │   ├── base.py         # Abstract interface
│       │   ├── pgvector.py
│       │   └── qdrant.py
│       │
│       ├── graph/
│       │   ├── client.py       # Neo4j async client
│       │   ├── schema.py
│       │   └── queries.py
│       │
│       ├── personas/
│       │   ├── discovery.py    # Hybrid clustering
│       │   ├── models.py
│       │   └── matching.py
│       │
│       ├── recommender/
│       │   ├── visual.py
│       │   ├── collaborative.py
│       │   ├── fusion.py       # RRF / Weighted merge
│       │   ├── reranker.py     # Cross-encoder
│       │   └── engine.py
│       │
│       ├── api/
│       │   ├── main.py
│       │   ├── dependencies.py
│       │   └── routers/
│       │
│       └── etl/
│           ├── hm_loader.py    # Polars ETL
│           ├── embedder.py
│           └── graph_loader.py
│
├── ui/
│   ├── app.py
│   └── pages/
│       ├── 1_visual_search.py
│       ├── 2_persona_demo.py
│       └── 3_graph_explorer.py
│
├── cli/
│   └── main.py                 # Typer CLI
│
├── n8n/
│   └── workflows/
│       ├── etl/
│       ├── realtime/
│       ├── personas/
│       └── integrations/
│
├── experimental/
│   ├── ranker.py               # Original experimental code
│   └── benchmarks/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│
├── scripts/
│   ├── setup_native.sh
│   ├── setup_qdrant.sh
│   └── create_sample_data.py
│
└── data/
    ├── sample/                 # Committed (small)
    ├── hm/                     # Gitignored (full)
    └── embeddings/             # Gitignored (generated)
```

---

## 6. Dependencies

```toml
[project]
name = "ragrec"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    # Core ML
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "sentence-transformers>=3.0.0",

    # Data processing (Polars, not Pandas!)
    "polars>=1.0.0",
    "pyarrow>=15.0.0",

    # Vector stores
    "pgvector>=0.3.0",
    "asyncpg>=0.29.0",
    "qdrant-client>=1.9.0",

    # Graph
    "neo4j>=5.20.0",

    # API & UI
    "fastapi>=0.111.0",
    "uvicorn>=0.30.0",
    "streamlit>=1.35.0",
    "typer>=0.12.0",

    # Image processing
    "pillow>=10.3.0",

    # Clustering
    "hdbscan>=0.8.33",
    "scikit-learn>=1.4.0",

    # Graph visualization
    "pyvis>=0.3.2",
    "networkx>=3.2.0",

    # Utilities
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "httpx>=0.27.0",
    "loguru>=0.7.0",

    # BM25 (for experimental ranker evaluation)
    "rank-bm25>=0.2.2",
]

[dependency-groups]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]
bench = [
    "memory-profiler>=0.61.0",
    "py-spy>=0.3.0",
]
```

---

## 7. CLI & API Design

### CLI Commands

```bash
# Data
uv run ragrec load-data ./data/hm --sample
uv run ragrec generate-embeddings --batch-size 64
uv run ragrec discover-personas --n-clusters 8

# Recommendations
uv run ragrec recommend "customer_abc123" --top-k 10
uv run ragrec similar ./product_image.jpg --top-k 10

# Benchmarking
uv run ragrec benchmark --vectorstore both --queries 1000

# Services
uv run ragrec serve --port 8000 --reload
uv run ragrec ui
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/recommend` | POST | Get personalized recommendations |
| `/api/v1/similar` | POST | Find visually similar products |
| `/api/v1/persona/{customer_id}` | GET | Get customer persona |
| `/api/v1/health` | GET | Health check |
| `/api/v1/benchmark/compare` | GET | Real-time vector store comparison |
| `/api/v1/internal/*` | POST | Internal endpoints for n8n |

---

## 8. Streamlit UI

### Pages

1. **Visual Search** (`1_visual_search.py`)
   - Upload product image
   - Display similar products grid
   - Category filtering

2. **Persona Explorer** (`2_persona_demo.py`)
   - Persona overview cards
   - Customer lookup
   - UMAP embedding visualization

3. **Graph Browser** (`3_graph_explorer.py`)
   - Product neighborhood queries
   - Customer journey visualization
   - Interactive pyvis graphs

---

## 9. n8n Workflow Automation

### Workflow Categories

```
n8n/workflows/
├── etl/
│   ├── daily_product_sync.json
│   ├── weekly_full_reindex.json
│   └── import_hm_dataset.json
│
├── realtime/
│   ├── new_product_webhook.json
│   ├── purchase_event.json
│   └── product_update_webhook.json
│
├── personas/
│   ├── daily_persona_refresh.json
│   ├── weekly_persona_discovery.json
│   └── persona_drift_alert.json
│
├── benchmark/
│   ├── weekly_vectorstore_bench.json
│   └── recommendation_quality.json
│
└── integrations/
    ├── export_to_crm.json
    ├── slack_alerts.json
    └── email_persona_report.json
```

### Key Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `new_product_webhook` | Webhook POST | Embed & index new products in real-time |
| `daily_persona_refresh` | Cron 3 AM | Update customer embeddings & persona assignments |
| `weekly_vectorstore_bench` | Cron Sunday 2 AM | Compare pgvector vs Qdrant performance |

---

## 10. Setup & Development

### Native Mac M2 Pro Setup

```bash
# Install system dependencies
brew install postgresql@17 pgvector neo4j n8n

# Start services
brew services start postgresql@17
brew services start neo4j

# Configure PostgreSQL
createdb ragrec
psql ragrec -c "CREATE EXTENSION IF NOT EXISTS vector"

# Install Python dependencies
uv sync

# Load sample data
uv run ragrec load-data ./data/sample --sample-only
```

### Makefile Targets

```makefile
setup          # Run native setup script
start          # Start PostgreSQL & Neo4j
stop           # Stop services
serve          # Start FastAPI with reload
ui             # Start Streamlit
test           # Run unit tests
lint           # Ruff check & format
benchmark      # Run vector store comparison
```

---

## 11. Implementation Phases

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| **0** | Foundation | Project scaffolding, services running, health endpoints |
| **1** | Core Pipeline | H&M loader, SigLIP embeddings, pgvector, basic visual search |
| **2** | GraphRAG | Neo4j schema, graph queries, graph-enhanced recommendations |
| **3** | Personas | Behavior embeddings, hybrid clustering, persona assignment |
| **4** | Full Integration | Fusion layer, complete API, Streamlit UI |
| **5** | n8n Automation | Workflows, webhooks, scheduled jobs |
| **6** | Benchmarking | Qdrant integration, performance comparison, Rust candidates |

### Rust Migration Candidates (Phase 6+)

| Component | Current | Rust Option | Expected Speedup |
|-----------|---------|-------------|------------------|
| Embedding inference | `transformers` | `candle` / `ort` | 2-5x |
| PCA compression | `sklearn` | `ndarray` + PyO3 | 3-10x |
| Batch vector ops | `numpy` | `faer` / `nalgebra` | 2-5x |
| Polars UDFs | Python | `pyo3-polars` | 5-20x |

---

## 12. Experimental Ranker Evaluation

The `experimental/ranker.py` file contains ideas to benchmark:

| Concept | Description | Evaluation |
|---------|-------------|------------|
| **Tri-Modal Fusion** | BM25 + Vector + Graph with weights | Adopt as core pattern |
| **Weighted Merge** | `w_bm25=0.34, w_vec=0.44, w_graph=0.22` | Benchmark vs RRF |
| **Cross-Encoder Rerank** | `ms-marco-MiniLM-L-6-v2` | Adopt for top-k refinement |
| **MinMax Normalization** | Score normalization before fusion | Benchmark vs rank-based |
| **EntityGraphIndex** | NetworkX co-occurrence graph | Compare vs Neo4j |

### Fusion Strategy Interface

```python
class FusionStrategy(ABC):
    @abstractmethod
    def fuse(self, visual_hits, persona_hits, graph_hits) -> pl.DataFrame: ...

class RRFFusion(FusionStrategy): ...      # Baseline
class WeightedFusion(FusionStrategy): ... # From experimental/ranker.py
class LearnedFusion(FusionStrategy): ...  # Future: learned weights
```

### Benchmark Matrix (to fill during implementation)

| Approach | nDCG@10 | P@5 | Latency (p95) | Keep? |
|----------|---------|-----|---------------|-------|
| RRF (k=60) | TBD | TBD | TBD | ? |
| Weighted merge | TBD | TBD | TBD | ? |
| + Cross-encoder | TBD | TBD | TBD | ? |

---

## Research References

### Key Papers (2024-2025)

- [GraphRAG Survey (arXiv 2501.00309)](https://arxiv.org/abs/2501.00309)
- [Improving Visual Recommendation on E-commerce (Mercari, RecSys 2025)](https://arxiv.org/html/2510.13359v1)
- [Multimodal Recommender Systems Survey (arXiv 2502.15711)](https://arxiv.org/html/2502.15711v1)
- [MerRec Dataset (KDD 2024)](https://arxiv.org/abs/2402.14230)

### Dataset

- **H&M Personalized Fashion Recommendations** (Kaggle)
  - 106K products, 1.37M customers, 31.8M transactions
  - Product images, customer age, transaction history

---

## Appendix: Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Embeddings** | SigLIP | Image + text encoding (768-dim) |
| **Vector Store** | pgvector + Qdrant | Similarity search (benchmarked) |
| **Graph** | Neo4j | Knowledge graph, personas, relationships |
| **API** | FastAPI | REST endpoints |
| **UI** | Streamlit | Demo interface |
| **CLI** | Typer | Batch operations |
| **Automation** | n8n | ETL, webhooks, scheduling |
| **Data** | Polars | All data processing (no Pandas) |
| **Runtime** | Native ARM | Mac M2 Pro optimized (Homebrew) |
