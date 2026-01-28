"""Create a larger sample focused on customers with multiple transactions."""

import polars as pl
from pathlib import Path
from rich.console import Console

console = Console()

def create_rich_sample(
    data_dir: Path,
    output_dir: Path,
    target_customers: int = 50_000,
    min_transactions: int = 3,
) -> None:
    """Create sample with customers who have multiple transactions.

    Args:
        data_dir: Path to full H&M CSV files
        output_dir: Path to output sample Parquet files
        target_customers: Target number of customers to include
        min_transactions: Minimum transactions per customer
    """
    console.print(f"[bold blue]Creating rich sample ({target_customers:,} customers, min {min_transactions} transactions)...[/bold blue]")

    # Read transactions to find active customers
    console.print("\n[1/5] Reading transactions...")
    transactions = pl.read_csv(
        data_dir / "transactions_train.csv",
        dtypes={
            "customer_id": pl.Utf8,
            "article_id": pl.Int64,
            "price": pl.Float64,
            "t_dat": pl.Utf8,
        },
    ).with_columns([
        pl.col("t_dat").str.to_date("%Y-%m-%d").alias("transaction_date")
    ])
    console.print(f"  Read {len(transactions):,} total transactions")

    # Find customers with multiple transactions
    console.print(f"\n[2/5] Finding customers with {min_transactions}+ transactions...")
    customer_txn_counts = (
        transactions
        .group_by("customer_id")
        .agg(pl.count("article_id").alias("txn_count"))
        .filter(pl.col("txn_count") >= min_transactions)
        .sort("txn_count", descending=True)
    )
    console.print(f"  Found {len(customer_txn_counts):,} customers with {min_transactions}+ transactions")

    # Sample target number of customers
    if len(customer_txn_counts) > target_customers:
        sampled_customers = customer_txn_counts.head(target_customers)
        console.print(f"  Sampled top {target_customers:,} most active customers")
    else:
        sampled_customers = customer_txn_counts
        console.print(f"  Using all {len(sampled_customers):,} customers")

    customer_ids = sampled_customers.select("customer_id")

    # Filter transactions for sampled customers
    console.print("\n[3/5] Filtering transactions...")
    filtered_transactions = transactions.join(customer_ids, on="customer_id", how="inner")
    console.print(f"  Kept {len(filtered_transactions):,} transactions")

    # Get unique articles from filtered transactions
    article_ids = filtered_transactions.select("article_id").unique()
    console.print(f"  Found {len(article_ids):,} unique articles")

    # Load and filter articles
    console.print("\n[4/5] Loading and filtering articles...")
    articles = pl.read_csv(
        data_dir / "articles.csv",
        dtypes={
            "article_id": pl.Int64,
            "product_code": pl.Int64,
            "prod_name": pl.Utf8,
            "product_type_no": pl.Int64,
            "product_type_name": pl.Utf8,
            "product_group_name": pl.Utf8,
            "graphical_appearance_no": pl.Int64,
            "graphical_appearance_name": pl.Utf8,
            "colour_group_code": pl.Int64,
            "colour_group_name": pl.Utf8,
            "perceived_colour_value_id": pl.Int64,
            "perceived_colour_value_name": pl.Utf8,
            "perceived_colour_master_id": pl.Int64,
            "perceived_colour_master_name": pl.Utf8,
            "department_no": pl.Int64,
            "department_name": pl.Utf8,
            "index_code": pl.Utf8,
            "index_name": pl.Utf8,
            "index_group_no": pl.Int64,
            "index_group_name": pl.Utf8,
            "section_no": pl.Int64,
            "section_name": pl.Utf8,
            "garment_group_no": pl.Int64,
            "garment_group_name": pl.Utf8,
            "detail_desc": pl.Utf8,
        },
    )
    filtered_articles = articles.join(article_ids, on="article_id", how="inner")
    console.print(f"  Filtered to {len(filtered_articles):,} articles")

    # Load and filter customers
    customers = pl.read_csv(
        data_dir / "customers.csv",
        dtypes={
            "customer_id": pl.Utf8,
            "FN": pl.Float64,
            "Active": pl.Float64,
            "club_member_status": pl.Utf8,
            "fashion_news_frequency": pl.Utf8,
            "age": pl.Int64,
            "postal_code": pl.Utf8,
        },
    )
    filtered_customers = customers.join(customer_ids, on="customer_id", how="inner")
    console.print(f"  Filtered to {len(filtered_customers):,} customers")

    # Save to Parquet
    console.print("\n[5/5] Saving to Parquet...")
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_articles.write_parquet(output_dir / "articles_sample.parquet")
    console.print(f"  ✓ Saved articles_sample.parquet")

    filtered_customers.write_parquet(output_dir / "customers_sample.parquet")
    console.print(f"  ✓ Saved customers_sample.parquet")

    # Rename transaction_date back to t_dat for consistency
    filtered_transactions = filtered_transactions.drop("transaction_date").with_columns([
        pl.col("t_dat")
    ])
    filtered_transactions.write_parquet(output_dir / "transactions_sample.parquet")
    console.print(f"  ✓ Saved transactions_sample.parquet")

    # Print statistics
    console.print("\n[bold green]✓ Sample creation complete![/bold green]")
    console.print(f"\nStatistics:")
    console.print(f"  Customers: {len(filtered_customers):,}")
    console.print(f"  Articles: {len(filtered_articles):,}")
    console.print(f"  Transactions: {len(filtered_transactions):,}")

    txn_stats = sampled_customers.select([
        pl.col("txn_count").mean().alias("avg"),
        pl.col("txn_count").median().alias("median"),
        pl.col("txn_count").max().alias("max"),
    ]).row(0)

    console.print(f"\nTransactions per customer:")
    console.print(f"  Average: {txn_stats[0]:.1f}")
    console.print(f"  Median: {txn_stats[1]:.0f}")
    console.print(f"  Max: {txn_stats[2]}")


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    target_customers = 50_000
    output_dir = Path("data/sample")

    if len(sys.argv) > 1:
        target_customers = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    data_dir = Path("data/hm")

    create_rich_sample(
        data_dir=data_dir,
        output_dir=output_dir,
        target_customers=target_customers,
        min_transactions=3,
    )
