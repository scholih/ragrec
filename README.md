# RagRec

**Multi-modal e-retail recommendation system** combining visual similarity, customer personas, and GraphRAG reasoning.

## Features

- 🎨 **Visual similarity** via SigLIP embeddings
- 👥 **Customer personas** via hybrid clustering (HDBSCAN + graph communities)
- 🧠 **GraphRAG reasoning** via Neo4j
- 🚀 **EU-sovereign, GDPR-compliant, ARM-native** (Mac M2 Pro optimized)

## Quick Start

### Prerequisites

- macOS with Homebrew
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# 1. Install native services (PostgreSQL, Neo4j)
make setup

# 2. Install Python dependencies
make dev-install

# 3. Copy environment template
cp .env.example .env
# Edit .env with your credentials

# 4. Set Neo4j password (first login)
open http://localhost:7474
```

### Running

```bash
# Start services
make start

# Run API server
make serve

# Run tests
make test

# Check service health
make health-check
```

## Architecture

```
src/ragrec/
├── embeddings/      # SigLIP visual embeddings
├── vectorstore/     # pgvector + Qdrant (abstract interface)
├── graph/           # Neo4j client and queries
├── personas/        # Customer persona discovery
├── recommender/     # Visual, collaborative, fusion
├── api/             # FastAPI application
└── etl/             # Data loading (Polars)
```

## Development

```bash
# Run linter
make lint

# Run type checker
make type-check

# Run all quality checks
make quality

# Format code
make format
```

## Documentation

- **Design Document**: `docs/plans/2025-01-28-ragrec-design.md`
- **Development Rules**: `MANIFESTO.md`
- **Project Context**: `CLAUDE.md`

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Visual Encoder | SigLIP | Proven at Mercari (+50% CTR) |
| Vector Store | pgvector + Qdrant | Benchmark comparison |
| Graph DB | Neo4j | Rich queries, existing experience |
| Data Processing | Polars | Faster than Pandas, Rust-based |
| Runtime | Homebrew native | ARM optimized, no Docker overhead |

## License

MIT
