"""Generate and store customer behavior embeddings."""

import asyncio
from datetime import datetime

import asyncpg
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ragrec.embeddings.sequence import BehaviorEncoder
from ragrec.etl.config import ETLConfig

console = Console()


async def generate_customer_behavior_embeddings(
    batch_size: int = 100,
    min_purchases: int = 1,
    recency_halflife_days: float = 30.0,
) -> dict[str, int]:
    """Generate behavior embeddings for all customers.

    Args:
        batch_size: Number of customers to process per batch
        min_purchases: Minimum purchases required (customers below this get zero embeddings)
        recency_halflife_days: Days for recency weight to decay to 0.5

    Returns:
        Dict with counts of customers processed
    """
    config = ETLConfig()
    encoder = BehaviorEncoder(
        embedding_dim=768,  # SigLIP dimension
        output_dim=256,
        recency_halflife_days=recency_halflife_days,
    )

    console.print("[bold blue]Generating customer behavior embeddings...[/bold blue]")
    console.print(f"  Recency halflife: {recency_halflife_days} days")
    console.print(f"  Output dimension: {encoder.output_dim}")

    # Connect to PostgreSQL
    pool = await asyncpg.create_pool(config.database_url)

    # Get all customers with their purchase history
    async with pool.acquire() as conn:
        # Count total customers
        total_customers = await conn.fetchval("SELECT COUNT(*) FROM customers")
        console.print(f"  Total customers: {total_customers:,}")

        # Get customers with purchase counts
        customers_with_purchases = await conn.fetch(
            """
            SELECT 
                c.customer_id,
                COUNT(t.id) AS purchase_count
            FROM customers c
            LEFT JOIN transactions t ON c.customer_id = t.customer_id
            GROUP BY c.customer_id
            ORDER BY c.customer_id
            """
        )

    # Process in batches
    reference_time = datetime.now()
    processed = 0
    embeddings_generated = 0
    zero_embeddings = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Processing customers...",
            total=len(customers_with_purchases),
        )

        for i in range(0, len(customers_with_purchases), batch_size):
            batch = customers_with_purchases[i : i + batch_size]

            # Fetch purchase data for batch
            customer_ids = [row["customer_id"] for row in batch]

            async with pool.acquire() as conn:
                # Get purchases with embeddings for this batch
                purchases = await conn.fetch(
                    """
                    SELECT 
                        t.customer_id,
                        t.article_id,
                        t.transaction_date,
                        pe.embedding::text as embedding_text
                    FROM transactions t
                    JOIN product_embeddings pe ON t.article_id = pe.article_id
                    WHERE t.customer_id = ANY($1)
                    ORDER BY t.customer_id, t.transaction_date
                    """,
                    customer_ids,
                )

            # Group purchases by customer
            customer_purchases = {}
            for purchase in purchases:
                customer_id = purchase["customer_id"]
                if customer_id not in customer_purchases:
                    customer_purchases[customer_id] = {"embeddings": [], "timestamps": []}

                # Parse vector text format: "[1.0,2.0,3.0]" -> numpy array
                embedding_text = purchase["embedding_text"]
                embedding_values = [float(x) for x in embedding_text.strip("[]").split(",")]
                embedding_array = np.array(embedding_values, dtype=np.float32)

                customer_purchases[customer_id]["embeddings"].append(embedding_array)
                # Convert date to datetime at midnight for compatibility
                transaction_date = purchase["transaction_date"]
                if hasattr(transaction_date, 'date'):
                    # Already datetime
                    timestamp = transaction_date
                else:
                    # Convert date to datetime
                    from datetime import time
                    timestamp = datetime.combine(transaction_date, time())
                customer_purchases[customer_id]["timestamps"].append(timestamp)

            # Generate embeddings for each customer in batch
            embeddings_to_store = []
            for customer_id in customer_ids:
                if customer_id in customer_purchases and len(customer_purchases[customer_id]["embeddings"]) >= min_purchases:
                    # Generate behavior embedding
                    embedding = encoder.encode_sequence(
                        product_embeddings=customer_purchases[customer_id]["embeddings"],
                        timestamps=customer_purchases[customer_id]["timestamps"],
                        reference_time=reference_time,
                    )
                    embeddings_generated += 1
                else:
                    # Zero embedding for customers with insufficient purchases
                    embedding = np.zeros(encoder.output_dim, dtype=np.float32)
                    zero_embeddings += 1

                embeddings_to_store.append({
                    "customer_id": customer_id,
                    "embedding": embedding.tolist(),
                })

            # Store embeddings in database
            async with pool.acquire() as conn:
                # Convert embeddings to string format for vector type
                for row in embeddings_to_store:
                    embedding_str = "[" + ",".join(map(str, row["embedding"])) + "]"
                    await conn.execute(
                        """
                        UPDATE customers
                        SET behavior_embedding = $1::vector
                        WHERE customer_id = $2
                        """,
                        embedding_str,
                        row["customer_id"],
                    )

            processed += len(batch)
            progress.update(task, advance=len(batch))

    await pool.close()

    console.print(f"\n[bold green]✓ Behavior embedding generation complete![/bold green]")
    console.print(f"  Processed: {processed:,} customers")
    console.print(f"  Generated: {embeddings_generated:,} behavior embeddings")
    console.print(f"  Zero embeddings: {zero_embeddings:,} (customers with <{min_purchases} purchases)")

    # Create index on behavior_embedding for similarity search
    console.print("\n[bold blue]Creating HNSW index on behavior embeddings...[/bold blue]")
    pool = await asyncpg.create_pool(config.database_url)
    async with pool.acquire() as conn:
        # Drop existing index if present
        await conn.execute("DROP INDEX IF EXISTS idx_customers_behavior_hnsw")

        # Create HNSW index
        await conn.execute(
            """
            CREATE INDEX idx_customers_behavior_hnsw 
            ON customers 
            USING hnsw (behavior_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )

    await pool.close()

    console.print("[bold green]✓ HNSW index created[/bold green]")

    return {
        "total_customers": processed,
        "embeddings_generated": embeddings_generated,
        "zero_embeddings": zero_embeddings,
    }


if __name__ == "__main__":
    asyncio.run(generate_customer_behavior_embeddings())
