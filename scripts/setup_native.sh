#!/usr/bin/env bash
set -euo pipefail

# RagRec Native Setup Script
# Installs PostgreSQL, pgvector, Neo4j, and n8n using Homebrew on macOS (ARM/Intel)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RagRec Native Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install from https://brew.sh"
    exit 1
fi

echo "✓ Homebrew detected"
echo ""

# Update Homebrew
echo "Updating Homebrew..."
brew update

# Install PostgreSQL 16
echo ""
echo "━━━ PostgreSQL + pgvector ━━━"
if brew list postgresql@16 &> /dev/null; then
    echo "✓ PostgreSQL 16 already installed"
else
    echo "Installing PostgreSQL 16..."
    brew install postgresql@16
fi

# Start PostgreSQL
echo "Starting PostgreSQL..."
brew services start postgresql@16

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if pg_isready -h localhost -p 5432 &> /dev/null; then
        echo "✓ PostgreSQL is ready"
        break
    fi
    sleep 1
done

# Create database and user
echo "Creating ragrec database and user..."
psql -U postgres -c "CREATE USER ragrec WITH PASSWORD 'changeme';" 2>/dev/null || echo "  User 'ragrec' already exists"
psql -U postgres -c "CREATE DATABASE ragrec OWNER ragrec;" 2>/dev/null || echo "  Database 'ragrec' already exists"

# Install pgvector extension
echo "Installing pgvector extension..."
if brew list pgvector &> /dev/null; then
    echo "✓ pgvector already installed"
else
    brew install pgvector
fi

# Enable pgvector in database
echo "Enabling pgvector extension in database..."
psql -U postgres -d ragrec -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo "✓ pgvector extension enabled"

# Verify pgvector
echo "Verifying pgvector installation..."
PGVECTOR_VERSION=$(psql -U postgres -d ragrec -tAc "SELECT extversion FROM pg_extension WHERE extname='vector';")
echo "✓ pgvector version: $PGVECTOR_VERSION"

# Install Neo4j
echo ""
echo "━━━ Neo4j Graph Database ━━━"
if brew list neo4j &> /dev/null; then
    echo "✓ Neo4j already installed"
else
    echo "Installing Neo4j..."
    brew install neo4j
fi

# Start Neo4j
echo "Starting Neo4j..."
brew services start neo4j

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo "✓ Neo4j is ready"
        break
    fi
    sleep 2
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Setup complete!"
echo ""
echo "Services installed and running:"
echo "  • PostgreSQL 16 with pgvector"
echo "  • Neo4j Community Edition"
echo ""
echo "Default credentials:"
echo "  • PostgreSQL: ragrec / changeme"
echo "  • Neo4j: neo4j / neo4j (change on first login)"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and update credentials"
echo "  2. Set Neo4j password: http://localhost:7474"
echo "  3. Run 'make dev-install' to install Python dependencies"
echo "  4. Run 'uv run ragrec serve' to start the API"
echo ""
echo "Service management:"
echo "  • make start  - Start all services"
echo "  • make stop   - Stop all services"
echo "  • make health-check - Verify services are running"
echo ""
