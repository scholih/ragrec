#!/bin/bash
set -e

# RagRec Sync to MacBook Script
# Syncs code, database, and sample data from M2 Pro to MacBook
# Usage: ./scripts/sync_to_macbook.sh [--full]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MACBOOK_HOST="macbook"
MACBOOK_PATH="~/demos/ragrec"
DUMP_FILE="/tmp/ragrec_dump_$(date +%Y%m%d_%H%M%S).sql"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== RagRec Sync to MacBook ===${NC}"
echo ""

# Check if --full flag is passed
FULL_SYNC=false
if [[ "$1" == "--full" ]]; then
    FULL_SYNC=true
    echo -e "${YELLOW}Full sync mode: Will transfer full H&M dataset (32GB)${NC}"
    echo ""
fi

# Step 1: Ensure all code is committed and pushed
echo -e "${BLUE}[1/7] Checking git status...${NC}"
cd "$PROJECT_ROOT"

if [[ -n $(git status --porcelain) ]]; then
    echo -e "${YELLOW}Warning: Uncommitted changes detected${NC}"
    git status --short
    read -p "Continue without committing? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted. Please commit your changes first.${NC}"
        exit 1
    fi
fi

echo "Pushing to GitHub..."
git push origin main || echo -e "${YELLOW}Push failed or already up-to-date${NC}"

# Step 2: Sync beads
echo ""
echo -e "${BLUE}[2/7] Syncing beads...${NC}"
~/.claude/plugins/cache/beads-marketplace/beads/0.42.0/bd_new sync || echo -e "${YELLOW}Beads sync skipped${NC}"

# Step 3: Create database dump
echo ""
echo -e "${BLUE}[3/7] Creating PostgreSQL dump...${NC}"
/opt/homebrew/opt/postgresql@17/bin/pg_dump ragrec > "$DUMP_FILE"
DUMP_SIZE=$(du -h "$DUMP_FILE" | cut -f1)
echo "Database dump created: $DUMP_FILE ($DUMP_SIZE)"

# Step 4: Check MacBook connectivity
echo ""
echo -e "${BLUE}[4/7] Checking MacBook connectivity...${NC}"
if ! ssh -o ConnectTimeout=5 "$MACBOOK_HOST" "echo 'Connected'" &>/dev/null; then
    echo -e "${RED}Error: Cannot connect to MacBook${NC}"
    echo "Please ensure:"
    echo "  1. MacBook is powered on and on same network"
    echo "  2. SSH is enabled: System Settings → General → Sharing → Remote Login"
    exit 1
fi
echo -e "${GREEN}✓ MacBook reachable${NC}"

# Step 5: Pull latest code on MacBook
echo ""
echo -e "${BLUE}[5/7] Updating code on MacBook...${NC}"
ssh "$MACBOOK_HOST" "cd $MACBOOK_PATH && git pull origin main"

# Step 6: Transfer database dump
echo ""
echo -e "${BLUE}[6/7] Transferring database dump...${NC}"
scp "$DUMP_FILE" "$MACBOOK_HOST:$MACBOOK_PATH/ragrec_dump_latest.sql"
echo -e "${GREEN}✓ Database dump transferred${NC}"

# Step 7: Optionally transfer full H&M dataset
if [[ "$FULL_SYNC" == true ]]; then
    echo ""
    echo -e "${BLUE}[7/7] Transferring full H&M dataset (32GB - this will take time)...${NC}"

    # Create data/hm directory on MacBook
    ssh "$MACBOOK_HOST" "mkdir -p $MACBOOK_PATH/data/hm"

    # Transfer H&M data using rsync (faster, resumable)
    rsync -avz --progress "$PROJECT_ROOT/data/hm/" "$MACBOOK_HOST:$MACBOOK_PATH/data/hm/"

    echo -e "${GREEN}✓ Full dataset transferred${NC}"
else
    echo ""
    echo -e "${BLUE}[7/7] Skipping full dataset transfer${NC}"
    echo -e "${YELLOW}Note: Use --full flag to transfer 32GB H&M dataset${NC}"
fi

# Cleanup local dump
rm "$DUMP_FILE"

# Instructions for MacBook
echo ""
echo -e "${GREEN}=== Sync Complete! ===${NC}"
echo ""
echo "To finish setup on MacBook, run:"
echo ""
echo -e "${BLUE}  ssh $MACBOOK_HOST${NC}"
echo -e "${BLUE}  cd $MACBOOK_PATH${NC}"
echo -e "${BLUE}  ./scripts/restore_database.sh${NC}"
echo ""
echo "Or run everything in one command:"
echo -e "${BLUE}  ssh $MACBOOK_HOST \"cd $MACBOOK_PATH && ./scripts/restore_database.sh\"${NC}"
echo ""
