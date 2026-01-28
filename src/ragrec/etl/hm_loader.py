"""H&M data loader using Polars and asyncpg."""

import asyncio
from pathlib import Path
from typing import Any

import asyncpg
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ragrec.etl.config import ETLConfig

console = Console()


def create_age_bracket(age: int | None) -> str | None:
    """Create age bracket from age value."""
    if age is None:
        return None
    if age < 20:
        return "under_20"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "60+"


class HMDataLoader:
    """Load H&M dataset into PostgreSQL using Polars."""

    def __init__(self, config: ETLConfig | None = None) -> None:
        """Initialize loader with configuration."""
        self.config = config or ETLConfig()
        self.pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "HMDataLoader":
        """Async context manager entry."""
        self.pool = await asyncpg.create_pool(
            self.config.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self.pool:
            await self.pool.close()

    async def initialize_schema(self) -> None:
        """Initialize database schema from SQL file."""
        console.print("[bold blue]Initializing database schema...[/bold blue]")

        schema_path = Path(__file__).parent / "sql" / "schema.sql"
        schema_sql = schema_path.read_text()

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)

        console.print("  ✓ Schema initialized")

    async def load_products(self, data_path: Path) -> int:
        """Load products from Parquet file."""
        console.print("\n[bold blue]Loading products...[/bold blue]")

        # Read with Polars
        df = pl.read_parquet(data_path / "articles_sample.parquet")
        console.print(f"  Read {len(df):,} products from Parquet")

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        # Prepare data for insertion
        records = df.to_dicts()

        # Batch insert
        batch_size = self.config.batch_size
        total_inserted = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Inserting products...", total=len(records))

            async with self.pool.acquire() as conn:
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]

                    # Build VALUES clause
                    values = []
                    for record in batch:
                        # Handle NULL values properly
                        vals = (
                            record.get("article_id"),
                            record.get("product_code"),
                            record.get("prod_name"),
                            record.get("product_type_no"),
                            record.get("product_type_name"),
                            record.get("product_group_name"),
                            record.get("graphical_appearance_no"),
                            record.get("graphical_appearance_name"),
                            record.get("colour_group_code"),
                            record.get("colour_group_name"),
                            record.get("perceived_colour_value_id"),
                            record.get("perceived_colour_value_name"),
                            record.get("perceived_colour_master_id"),
                            record.get("perceived_colour_master_name"),
                            record.get("department_no"),
                            record.get("department_name"),
                            record.get("index_code"),
                            record.get("index_name"),
                            record.get("index_group_no"),
                            record.get("index_group_name"),
                            record.get("section_no"),
                            record.get("section_name"),
                            record.get("garment_group_no"),
                            record.get("garment_group_name"),
                            record.get("detail_desc"),
                        )
                        values.append(vals)

                    # Execute batch insert
                    await conn.executemany(
                        """
                        INSERT INTO products (
                            article_id, product_code, prod_name, product_type_no,
                            product_type_name, product_group_name, graphical_appearance_no,
                            graphical_appearance_name, colour_group_code, colour_group_name,
                            perceived_colour_value_id, perceived_colour_value_name,
                            perceived_colour_master_id, perceived_colour_master_name,
                            department_no, department_name, index_code, index_name,
                            index_group_no, index_group_name, section_no, section_name,
                            garment_group_no, garment_group_name, detail_desc
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
                        ON CONFLICT (article_id) DO NOTHING
                        """,
                        values,
                    )

                    total_inserted += len(batch)
                    progress.update(task, advance=len(batch))

        console.print(f"  ✓ Inserted {total_inserted:,} products")
        return total_inserted

    async def load_customers(self, data_path: Path) -> int:
        """Load customers from Parquet file."""
        console.print("\n[bold blue]Loading customers...[/bold blue]")

        # Read with Polars
        df = pl.read_parquet(data_path / "customers_sample.parquet")
        console.print(f"  Read {len(df):,} customers from Parquet")

        # Add age_bracket column
        df = df.with_columns(
            pl.col("age")
            .map_elements(create_age_bracket, return_dtype=pl.Utf8)
            .alias("age_bracket")
        )

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        # Prepare data for insertion
        records = df.to_dicts()

        # Batch insert
        batch_size = self.config.batch_size
        total_inserted = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Inserting customers...", total=len(records))

            async with self.pool.acquire() as conn:
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]

                    values = []
                    for record in batch:
                        vals = (
                            record.get("customer_id"),
                            record.get("FN"),
                            record.get("Active"),
                            record.get("club_member_status"),
                            record.get("fashion_news_frequency"),
                            record.get("age"),
                            record.get("age_bracket"),
                            record.get("postal_code"),
                        )
                        values.append(vals)

                    await conn.executemany(
                        """
                        INSERT INTO customers (
                            customer_id, fn, active, club_member_status,
                            fashion_news_frequency, age, age_bracket, postal_code
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (customer_id) DO NOTHING
                        """,
                        values,
                    )

                    total_inserted += len(batch)
                    progress.update(task, advance=len(batch))

        console.print(f"  ✓ Inserted {total_inserted:,} customers")
        return total_inserted

    async def load_transactions(self, data_path: Path) -> int:
        """Load transactions from Parquet file."""
        console.print("\n[bold blue]Loading transactions...[/bold blue]")

        # Read with Polars
        df = pl.read_parquet(data_path / "transactions_sample.parquet")
        console.print(f"  Read {len(df):,} transactions from Parquet")

        # Convert date column to proper format
        df = df.with_columns(pl.col("t_dat").str.strptime(pl.Date, "%Y-%m-%d"))

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        # Prepare data for insertion
        records = df.to_dicts()

        # Batch insert
        batch_size = self.config.batch_size
        total_inserted = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Inserting transactions...", total=len(records))

            async with self.pool.acquire() as conn:
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]

                    values = []
                    for record in batch:
                        vals = (
                            record.get("t_dat"),
                            record.get("customer_id"),
                            record.get("article_id"),
                            record.get("price"),
                        )
                        values.append(vals)

                    await conn.executemany(
                        """
                        INSERT INTO transactions (
                            transaction_date, customer_id, article_id, price
                        ) VALUES ($1, $2, $3, $4)
                        """,
                        values,
                    )

                    total_inserted += len(batch)
                    progress.update(task, advance=len(batch))

        console.print(f"  ✓ Inserted {total_inserted:,} transactions")
        return total_inserted

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM data_stats")
            return dict(row) if row else {}

    async def clear_data(self) -> None:
        """Clear all data from tables (for testing)."""
        console.print("[bold yellow]Clearing all data...[/bold yellow]")

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE transactions, customers, products CASCADE")

        console.print("  ✓ Data cleared")


async def load_hm_data(data_path: Path, clear_first: bool = False) -> None:
    """Load H&M dataset into PostgreSQL.

    Args:
        data_path: Path to directory containing Parquet files
        clear_first: Whether to clear existing data first
    """
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("  H&M Data Loader (Polars → PostgreSQL)")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"\nData path: {data_path.absolute()}")

    async with HMDataLoader() as loader:
        # Initialize schema
        await loader.initialize_schema()

        # Clear data if requested
        if clear_first:
            await loader.clear_data()

        # Load data
        products_count = await loader.load_products(data_path)
        customers_count = await loader.load_customers(data_path)
        transactions_count = await loader.load_transactions(data_path)

        # Get final stats
        stats = await loader.get_stats()

        # Summary
        console.print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        console.print("  ✅ Data Load Complete!")
        console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        console.print(f"\nLoaded this session:")
        console.print(f"  • Products: {products_count:,}")
        console.print(f"  • Customers: {customers_count:,}")
        console.print(f"  • Transactions: {transactions_count:,}")

        console.print(f"\nDatabase totals:")
        console.print(f"  • Products: {stats.get('products_count', 0):,}")
        console.print(f"  • Customers: {stats.get('customers_count', 0):,}")
        console.print(f"  • Transactions: {stats.get('transactions_count', 0):,}")
        console.print(
            f"  • Transaction date range: {stats.get('earliest_transaction')} to {stats.get('latest_transaction')}"
        )
        console.print()


if __name__ == "__main__":
    # For testing
    asyncio.run(load_hm_data(Path("data/sample"), clear_first=True))
