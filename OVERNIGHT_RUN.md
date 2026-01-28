# Overnight Embedding Generation - 50K Sample

**Started:** 2026-01-28 evening
**Process ID:** 63414
**Log file:** `/tmp/ragrec-embeddings-50k.log`

## What's Running

Generating SigLIP embeddings for **96,823 products** from the 50K customer rich sample.

**Dataset stats:**
- Customers: 50,000 (top most active)
- Products: 96,823
- Transactions: 8,664,680
- Avg transactions/customer: **173** (vs 1 in original 5K sample)

**Embedding parameters:**
- Model: google/siglip-base-patch16-224
- Batch size: 128 (optimized for M2 Pro MPS)
- Embedding dimension: 768

**Estimated completion:**
- Batch preparation: ~15 minutes (checking 96K image files)
- Embedding generation: ~90-120 minutes (756 batches @ ~2 batches/sec)
- **Total: ~2 hours** from start

## Check Progress

```bash
# Check if still running
ps -p 63414

# Monitor progress (Ctrl+C to exit)
tail -f /tmp/ragrec-embeddings-50k.log

# Check database progress
uv run python -c "
import asyncio
import asyncpg

async def check():
    conn = await asyncpg.connect('postgresql://ragrec:ragrec123@localhost:5432/ragrec')

    # Check embeddings table
    try:
        count = await conn.fetchval('SELECT COUNT(*) FROM embeddings')
        total = await conn.fetchval('SELECT COUNT(*) FROM products')
        print(f'Embeddings generated: {count:,} / {total:,} ({count/total*100:.1f}%)')
    except:
        print('Embeddings table not created yet (still in preparation phase)')

    await conn.close()

asyncio.run(check())
"
```

## Next Steps (Tomorrow Morning)

Once embeddings are complete:

### 1. Generate Behavior Embeddings
```bash
uv run ragrec generate-behavior-embeddings
```
Expected: ~20 minutes for 50K customers

### 2. Load Graph into Neo4j
```bash
uv run ragrec load-graph --clear-first --top-k-similar=5
```
Expected: ~15 minutes

### 3. Discover Personas
```bash
uv run ragrec discover-personas --min-cluster-size=120 --min-samples=15
```
Expected result with rich behavioral data:
- **6-10 distinct personas** (vs 9 in 5K sample)
- **<30% uncategorized** (vs 67% in 5K sample!)
- Much more interpretable characteristics
- Meaningful age and category segmentation

### 4. Verify Results
```bash
# Check persona distribution
uv run python -c "
import asyncio
from ragrec.graph.client import Neo4jClient

async def check():
    async with Neo4jClient() as client:
        result = await client.execute_read('''
            MATCH (p:Persona)
            WHERE p.id <> 'persona_uncategorized'
            RETURN p.id AS id, p.name AS name, p.size AS size
            ORDER BY p.size DESC
        ''')

        total = sum(p['size'] for p in result)
        uncategorized = await client.execute_read('''
            MATCH (p:Persona {id: 'persona_uncategorized'})
            RETURN p.size AS size
        ''')

        print(f'Real personas: {len(result)}')
        print(f'Clustered customers: {total:,} ({total/50000*100:.1f}%)')
        if uncategorized:
            print(f'Uncategorized: {uncategorized[0][\"size\"]:,} ({uncategorized[0][\"size\"]/50000*100:.1f}%)')
        print()

        for p in result:
            pct = p['size'] / total * 100 if total > 0 else 0
            print(f'{p[\"name\"]:30s} {p[\"size\"]:5d} ({pct:5.1f}%)')

asyncio.run(check())
"
```

### 5. Run Integration Tests
```bash
uv run pytest tests/integration/ -v
```

### 6. Commit & Push
```bash
git add data/sample/
git commit -m "Update to 50K rich sample with full embeddings

- 50,000 customers (173 avg transactions each)
- 96,823 products with SigLIP embeddings
- 8.6M transactions
- Expect much better persona clustering

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
"
git push
```

## Troubleshooting

### If process died overnight
```bash
# Check if process is running
ps -p 63414

# If not running, check where it stopped
tail -50 /tmp/ragrec-embeddings-50k.log

# Restart from where it left off (will skip existing embeddings)
nohup uv run ragrec generate-embeddings data/hm/images --model=google/siglip-base-patch16-224 --batch-size=128 > /tmp/ragrec-embeddings-50k-restart.log 2>&1 &
```

### If embeddings table doesn't exist
The generator creates the embeddings table on first batch. If you only see "Fetching products..." in the log, it's still in the batch preparation phase (checking 96K image files exist).

## Expected Results

With rich behavioral data (173 avg transactions vs 1):
- HDBSCAN will find **more meaningful clusters**
- Personas will have **distinct behavioral patterns**
- **Much lower** uncategorized rate (expect 20-30% vs 67%)
- Clear segmentation by **age**, **category preferences**, **purchase frequency**

This is the data quality needed for production-grade persona discovery!
