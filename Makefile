.PHONY: help install dev-install test lint type-check format health-check setup start stop clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv pip install -e .

dev-install: ## Install development dependencies
	uv pip install -e ".[dev]"

test: ## Run tests
	uv run pytest tests/unit -v

test-cov: ## Run tests with coverage
	uv run pytest tests/unit -v --cov=ragrec --cov-report=html --cov-report=term

lint: ## Run linter (ruff check)
	uv run ruff check src/ tests/

lint-fix: ## Run linter with auto-fix
	uv run ruff check --fix src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/

type-check: ## Run type checker (mypy)
	uv run mypy src/ragrec

quality: lint type-check ## Run all quality checks (lint + type-check)

serve: ## Start FastAPI server
	uv run ragrec serve --reload

ui: ## Start Streamlit UI (not yet implemented)
	uv run ragrec ui

health-check: ## Check health of all services
	@echo "Checking PostgreSQL..."
	@pg_isready -h localhost -p 5432 && echo "✓ PostgreSQL is running" || echo "✗ PostgreSQL is not running"
	@echo ""
	@echo "Checking Neo4j..."
	@curl -s http://localhost:7474 > /dev/null && echo "✓ Neo4j is running" || echo "✗ Neo4j is not running"
	@echo ""
	@echo "Checking API health endpoint..."
	@curl -s http://localhost:8000/api/v1/health > /dev/null && echo "✓ API is running" || echo "✗ API is not running"

setup: ## Run native setup script (install PostgreSQL, Neo4j, etc.)
	@bash scripts/setup_native.sh

start: ## Start PostgreSQL and Neo4j services
	@echo "Starting PostgreSQL..."
	@brew services start postgresql@16 || echo "PostgreSQL already running"
	@echo "Starting Neo4j..."
	@brew services start neo4j || echo "Neo4j already running"
	@echo ""
	@echo "Services started. Run 'make health-check' to verify."

stop: ## Stop PostgreSQL and Neo4j services
	@echo "Stopping PostgreSQL..."
	@brew services stop postgresql@16
	@echo "Stopping Neo4j..."
	@brew services stop neo4j
	@echo "Services stopped."

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

reset-db: ## Reset PostgreSQL database (WARNING: deletes all data)
	@echo "⚠️  This will delete all data in the 'ragrec' database!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		psql -U postgres -c "DROP DATABASE IF EXISTS ragrec;"; \
		psql -U postgres -c "CREATE DATABASE ragrec;"; \
		psql -U postgres -d ragrec -c "CREATE EXTENSION IF NOT EXISTS vector;"; \
		echo "✓ Database reset complete"; \
	fi
