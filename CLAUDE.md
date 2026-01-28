# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

RagRec is a multi-modal e-retail recommendation system combining:
- **Visual similarity** via SigLIP embeddings
- **Customer personas** via hybrid clustering (HDBSCAN + graph communities)
- **GraphRAG reasoning** via Neo4j

**Target**: EU-sovereign, GDPR-compliant, ARM-native (Mac M2 Pro)

## Quick Reference

### Design Document
Full architecture specification: `docs/plans/2025-01-28-ragrec-design.md`

### Issue Tracker (Beads)
```bash
# List all issues
~/.claude/plugins/cache/beads-marketplace/beads/0.42.0/bd_new list

# Show ready work (no blockers)
~/.claude/plugins/cache/beads-marketplace/beads/0.42.0/bd_new ready

# Show specific issue
~/.claude/plugins/cache/beads-marketplace/beads/0.42.0/bd_new show <issue-id>

# Mark issue in progress
~/.claude/plugins/cache/beads-marketplace/beads/0.42.0/bd_new update <issue-id> --status in-progress

# Close issue
~/.claude/plugins/cache/beads-marketplace/beads/0.42.0/bd_new close <issue-id>
```

### Current Status
- 20 issues planned (Phases 0-6)
- Ready to start: `ragrec-zvc` (Phase 0: Foundation)

## Key Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Visual Encoder | SigLIP | Proven at Mercari (+50% CTR) |
| Vector Store | pgvector + Qdrant | Benchmark comparison |
| Graph DB | Neo4j | Rich queries, existing experience |
| Data Processing | Polars | Faster than Pandas, Rust-based |
| Runtime | Homebrew native | ARM optimized, no Docker overhead |
| Personas | HDBSCAN + Louvain | Hybrid embedding + graph clustering |

## Development Standards

### Always Use
- **Polars** instead of Pandas for data processing
- **uv run** for Python execution
- **Abstract interfaces** in `base.py` files (for future Rust swap)
- **Async** for all I/O operations

### Project Structure
```
src/ragrec/          # Main package
├── embeddings/      # SigLIP, sequence encoders
├── vectorstore/     # pgvector, Qdrant (abstract interface)
├── graph/           # Neo4j client and queries
├── personas/        # Discovery and matching
├── recommender/     # Visual, collaborative, fusion
├── api/             # FastAPI application
└── etl/             # Data loading (Polars)

ui/                  # Streamlit demo
cli/                 # Typer CLI
n8n/workflows/       # Automation workflows
experimental/        # Ranker evaluation code
```

### Commands
```bash
# Development
uv run ragrec serve --reload    # Start API
uv run ragrec ui                # Start Streamlit
uv run pytest tests/unit -v     # Run tests

# Quality
uv run ruff check --fix src/
uv run ruff format src/
uv run mypy src/ragrec
```

## Data

- **Sample data**: `data/sample/` (committed, ~1000 products)
- **Full H&M dataset**: `data/hm/` (gitignored, download from Kaggle)
- **Embeddings**: `data/embeddings/` (gitignored, generated)

## Experimental Code

`experimental/ranker.py` contains ideas to benchmark:
- Tri-modal fusion (BM25 + Vector + Graph)
- Weighted merge vs RRF
- Cross-encoder reranking

Evaluate in Phase 6.1 (`ragrec-z14`).
