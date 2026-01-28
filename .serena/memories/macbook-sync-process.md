# MacBook Sync Process

## Overview

RagRec development happens on M2 Pro but can be synced to MacBook for testing/portability.

**Key principle:** Don't transfer the 32GB H&M dataset unless absolutely needed. Sample data (29MB) + database dump (11MB) is sufficient for most work.

## Quick Sync (Code + Database)

```bash
# From M2 Pro
./scripts/sync_to_macbook.sh

# Then on MacBook (or run remotely)
ssh macbook "cd ~/demos/ragrec && ./scripts/restore_database.sh"
```

## Full Sync (Including 32GB Dataset)

```bash
# From M2 Pro
./scripts/sync_to_macbook.sh --full

# Then on MacBook
ssh macbook "cd ~/demos/ragrec && ./scripts/restore_database.sh"
```

## What Gets Synced

### Quick Sync (Default)
- ✅ All code (via git pull)
- ✅ Database dump (11MB with 1000 products + embeddings)
- ✅ Sample data (already in git - 29MB with 100 images)
- ✅ Beads issues
- ❌ Full H&M dataset (32GB)

### Full Sync (--full flag)
- Everything above PLUS:
- ✅ Full H&M dataset (32GB - articles.csv, customers.csv, transactions.csv, all images)

## MacBook Setup Requirements

### First Time Setup
```bash
# Install PostgreSQL 17 + pgvector
brew install postgresql@17 pgvector

# Start PostgreSQL
brew services start postgresql@17

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
cd ~/demos
git clone https://github.com/scholih/ragrec.git
cd ragrec

# Install dependencies
uv sync

# Add dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov
```

### Environment Configuration
Create `.env` file (minimal - only DATABASE_URL):
```bash
cd ~/demos/ragrec
cat > .env << 'EOF'
# Database Configuration
DATABASE_URL=postgresql://hjscholing@localhost/ragrec
EOF
```

**IMPORTANT:** Don't add `API_HOST` or `API_PORT` to `.env` - ETLConfig only expects `DATABASE_URL`

## SSH Configuration

MacBook must be reachable via SSH:
```bash
# On M2 Pro, ~/.ssh/config should have:
Host macbook
    HostName HJs-MBP-3.fritz.box
    User hjscholing
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Enable Remote Login on MacBook:
- System Settings → General → Sharing → Remote Login (ON)

## Verification

After sync + restore, verify on MacBook:
```bash
ssh macbook
cd ~/demos/ragrec

# Test database connection
uv run pytest tests/integration/test_pgvector.py::test_pgvector_health_check -v

# Run all tests
uv run pytest tests/unit tests/integration/test_pgvector.py -v

# Should see: 14 passed
```

## Common Issues

### Issue: "role ragrec does not exist"
- **Cause:** Database dump includes ownership commands
- **Fix:** Ignore - not critical, data still loads correctly

### Issue: Tests fail with "Extra inputs are not permitted" for api_host/api_port
- **Cause:** `.env` file has fields ETLConfig doesn't expect
- **Fix:** `.env` should ONLY have `DATABASE_URL`, nothing else

### Issue: SSH connection fails
- **Cause:** MacBook not on network or Remote Login disabled
- **Fix:** 
  1. Ping MacBook: `ping HJs-MBP-3.fritz.box`
  2. Enable Remote Login on MacBook
  3. Re-add SSH host key: `ssh-keygen -R HJs-MBP-3.fritz.box`

### Issue: PostgreSQL not starting
- **Cause:** PostgreSQL service not running
- **Fix:** `brew services start postgresql@17`

## Performance Notes

- **Quick sync time:** ~30 seconds (git pull + 11MB dump transfer)
- **Full sync time:** ~15-30 minutes (32GB dataset transfer via rsync)
- **Database restore:** ~5 seconds

## When to Use Full Sync

Only use `--full` flag when:
- Need to regenerate embeddings from full dataset
- Testing ETL pipeline with real 31.8M transactions
- Benchmarking performance on full data

For development/testing of Phase 1-3 features, quick sync is sufficient.
