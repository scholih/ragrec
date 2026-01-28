# RagRec Development Manifesto

**Version:** 1.0.0
**Last Updated:** 2026-01-28

## The Golden Rule

**No merge to main without: Tests Pass → Validation → Explicit Confirmation**

## Core Principles

### 1. Development Workflow
- **All feature work in git worktrees** - Never develop directly on main
- **Worktrees inside repo** - Use `.worktrees/` directory, never parent paths
- **One feature per worktree** - Keep changes isolated and focused
- **Clean state before merge** - No uncommitted changes, all tests passing

### 2. Testing Requirements
- **Unit tests required** - All new code must have unit tests
- **Integration tests for APIs** - FastAPI endpoints must have integration tests
- **Benchmark before merge** - Vector search and ranking changes require benchmarks
- **No broken tests** - Fix or skip explicitly, never commit with failures

### 3. Code Quality Gates
```bash
# All of these must pass before merge consideration:
uv run ruff check --fix src/
uv run ruff format src/
uv run mypy src/ragrec
uv run pytest tests/unit -v
```

### 4. Data Handling Standards
- **Sample data only in repo** - Full datasets stay in gitignored `data/hm/`
- **Embeddings are artifacts** - Never commit to git, regenerate as needed
- **Polars for ETL** - Use Polars instead of Pandas for all data processing
- **Abstract data sources** - Use interfaces to swap implementations later

### 5. Service Management
- **Native services via Homebrew** - PostgreSQL, Neo4j, n8n installed natively
- **No Docker overhead** - ARM-optimized native binaries for M-series Macs
- **Health checks mandatory** - All services must expose health endpoints
- **Graceful degradation** - System works with partial service availability

### 6. Async-First Development
- **All I/O is async** - Database, API calls, file operations
- **Use async context managers** - Proper resource cleanup
- **Connection pooling** - Reuse connections, don't create per-request
- **Timeouts everywhere** - All async operations must have timeouts

### 7. Interface-Driven Design
- **Abstract base classes** - Define interfaces in `base.py` files
- **Dependency injection** - Pass dependencies, don't instantiate in code
- **Swap implementations** - Ready to replace Python with Rust where needed
- **Type hints required** - All function parameters and returns typed

### 8. Experimental Code Policy
- **Sandbox in `experimental/`** - Prototype freely, no quality gates
- **Benchmark before promoting** - Prove performance before moving to `src/`
- **Document trade-offs** - Capture why you chose one approach over another
- **Commit experiments** - Keep history of what was tried and why

## Prohibited Actions

❌ **Never merge without testing**
❌ **Never commit broken tests**
❌ **Never use Pandas** - Use Polars
❌ **Never use plain `python`** - Use `uv run python`
❌ **Never commit secrets** - Use `.env` (gitignored)
❌ **Never commit large datasets** - Use `data/sample/` only
❌ **Never skip type hints** - Type all public APIs
❌ **Never block async operations** - Use `await` properly

## Pre-Merge Checklist

Before requesting merge to main:

```bash
# 1. Run quality checks
make lint
make type-check
make test

# 2. Verify services
make health-check

# 3. Check git state
git status  # Must be clean

# 4. Run benchmarks (if applicable)
uv run python experimental/benchmark.py

# 5. Analyze results
# Review test output, benchmark numbers, any warnings

# 6. Get explicit approval
# Ask user: "Tests pass, validation complete. Merge to main?"
# Wait for explicit "yes" - never assume approval
```

## Version Bumping

Follow semantic versioning:
- **MAJOR**: Breaking API changes, architectural rewrites
- **MINOR**: New features, backward-compatible additions
- **PATCH**: Bug fixes, performance improvements

Document version bumps in commit messages:
```
feat: Add GraphRAG persona matching (bumps MINOR)
fix: Correct SigLIP embedding dimension (bumps PATCH)
feat!: Replace pgvector with Qdrant (bumps MAJOR)
```

## Performance Standards

### Vector Search
- **<100ms p50** - Median latency for similarity search
- **<500ms p99** - 99th percentile must be under 500ms
- **1000+ QPS** - Support at least 1000 queries/second

### API Response Times
- **<200ms p50** - Median API latency
- **<1s p99** - 99th percentile under 1 second
- **Health endpoint <10ms** - Fast health checks

### Resource Usage
- **<2GB RAM** - Per service at idle
- **<4GB RAM** - Per service under load
- **<50% CPU** - Average CPU usage under load

## Dependency Philosophy

### Prefer
- **Polars** over Pandas (faster, Rust-based)
- **asyncpg** over psycopg2 (async, faster)
- **httpx** over requests (async support)
- **orjson** over json (faster JSON serialization)
- **uvloop** over default event loop (faster async)

### Avoid
- Heavy frameworks when lightweight works
- Dependencies with C extensions (prefer Rust)
- Deprecated packages (check PyPI status)
- Packages without type hints

## Security Requirements

- **Environment variables for secrets** - Never hardcode credentials
- **Validate all inputs** - Especially from API requests
- **GDPR compliance** - EU data residency, right to deletion
- **Rate limiting** - Protect endpoints from abuse
- **Audit logging** - Track sensitive operations

## Documentation Standards

- **Docstrings for public APIs** - Google style docstrings
- **Type hints over comments** - Let types document parameters
- **README per module** - Explain module purpose and usage
- **Architecture decision records** - Document major choices
- **Keep CLAUDE.md updated** - Project context for AI assistants

## Violation = Broken Trust

This manifesto exists because these rules prevent real problems:
- Broken tests ship bugs to users
- Missing type hints make refactoring dangerous
- Pandas is 10-100x slower than Polars on large datasets
- Blocking I/O kills throughput under load
- Missing validation lets bad data corrupt the system

**Follow the manifesto. It's here because we learned the hard way.**

---

*Questions about this manifesto? Update this file and commit the change.*
