#!/bin/bash
set -e

# RagRec Database Restore Script (MacBook)
# Restores database from dump created by sync_to_macbook.sh
# Usage: ./scripts/restore_database.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== RagRec Database Restore ===${NC}"
echo ""

# Add PostgreSQL to PATH
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"

# Check if dump file exists
DUMP_FILE="$PROJECT_ROOT/ragrec_dump_latest.sql"
if [[ ! -f "$DUMP_FILE" ]]; then
    echo -e "${RED}Error: Database dump not found at $DUMP_FILE${NC}"
    echo "Please run sync_to_macbook.sh from your M2 Pro first"
    exit 1
fi

DUMP_SIZE=$(du -h "$DUMP_FILE" | cut -f1)
echo "Found database dump: $DUMP_SIZE"
echo ""

# Check if PostgreSQL is running
echo -e "${BLUE}[1/4] Checking PostgreSQL...${NC}"
if ! pg_isready -h localhost &>/dev/null; then
    echo "Starting PostgreSQL..."
    brew services start postgresql@17
    sleep 3

    if ! pg_isready -h localhost &>/dev/null; then
        echo -e "${RED}Error: PostgreSQL failed to start${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ PostgreSQL running${NC}"

# Drop and recreate database (fresh start)
echo ""
echo -e "${BLUE}[2/4] Recreating database...${NC}"
dropdb ragrec 2>/dev/null || echo "Database didn't exist"
createdb ragrec
echo -e "${GREEN}✓ Database created${NC}"

# Restore from dump
echo ""
echo -e "${BLUE}[3/4] Restoring database from dump...${NC}"
psql ragrec < "$DUMP_FILE" > /tmp/restore_output.log 2>&1

# Check for errors (ignore role ownership warnings)
if grep -i "error" /tmp/restore_output.log | grep -v "role.*does not exist" > /dev/null; then
    echo -e "${YELLOW}Warning: Some errors occurred during restore${NC}"
    echo "Check /tmp/restore_output.log for details"
else
    echo -e "${GREEN}✓ Database restored successfully${NC}"
fi

# Verify data
echo ""
echo -e "${BLUE}[4/4] Verifying data...${NC}"
PRODUCT_COUNT=$(psql ragrec -t -c "SELECT COUNT(*) FROM products;" | xargs)
EMBEDDING_COUNT=$(psql ragrec -t -c "SELECT COUNT(*) FROM product_embeddings;" | xargs)
CUSTOMER_COUNT=$(psql ragrec -t -c "SELECT COUNT(*) FROM customers;" | xargs)
TRANSACTION_COUNT=$(psql ragrec -t -c "SELECT COUNT(*) FROM transactions;" | xargs)

echo "Database contents:"
echo "  Products: $PRODUCT_COUNT"
echo "  Embeddings: $EMBEDDING_COUNT"
echo "  Customers: $CUSTOMER_COUNT"
echo "  Transactions: $TRANSACTION_COUNT"

if [[ "$PRODUCT_COUNT" -gt 0 && "$EMBEDDING_COUNT" -gt 0 ]]; then
    echo -e "${GREEN}✓ Data verified${NC}"
else
    echo -e "${RED}Error: Data verification failed${NC}"
    exit 1
fi

# Cleanup
rm /tmp/restore_output.log

echo ""
echo -e "${GREEN}=== Restore Complete! ===${NC}"
echo ""
echo "You can now run:"
echo -e "${BLUE}  uv run pytest tests/unit tests/integration/test_pgvector.py -v${NC}"
echo ""
