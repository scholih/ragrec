"""Load Amazon sample data into PostgreSQL.

Converts sampled Amazon JSONL data to Parquet format compatible with RagRec schema.
"""

import gzip
import json
from pathlib import Path
from datetime import datetime

import polars as pl
from rich.console import Console

console = Console()


def load_amazon_products(data_dir: Path, output_dir: Path) -> None:
    """Load Amazon product metadata into products table format.

    Args:
        data_dir: Directory containing Amazon JSONL.gz files
        output_dir: Output directory for Parquet files
    """
    console.print("[bold blue]Loading Amazon product data...[/bold blue]\n")

    all_products = []

    categories = [
        "All_Beauty",
        "Clothing_Shoes_and_Jewelry",
        "Electronics",
        "Books",
        "Sports_and_Outdoors",
    ]

    for category in categories:
        console.print(f"[cyan]{category}[/cyan]")

        meta_file = data_dir / f"meta_{category}.jsonl.gz"
        sampled_products_file = data_dir / f"sampled_products_{category}.txt"

        if not meta_file.exists() or not sampled_products_file.exists():
            console.print(f"  ⚠ Skipping (files not found)")
            continue

        # Load sampled product IDs
        with open(sampled_products_file) as f:
            sampled_asins = set(line.strip() for line in f)

        console.print(f"  Loading {len(sampled_asins):,} sampled products...")

        # Load metadata for sampled products
        products_loaded = 0

        with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    product = json.loads(line)
                    parent_asin = product.get('parent_asin')

                    if parent_asin not in sampled_asins:
                        continue

                    # Extract relevant fields
                    product_data = {
                        'article_id': parent_asin,
                        'product_code': parent_asin[:10],  # Truncate for code
                        'prod_name': product.get('title', '')[:255],  # Limit length
                        'main_category': category,
                        'category': product.get('categories', [''])[0] if product.get('categories') else '',
                        'brand': product.get('brand', '')[:50],
                        'price': product.get('price', 0.0) if product.get('price') else 0.0,
                        'rating': product.get('average_rating', 0.0) if product.get('average_rating') else 0.0,
                        'rating_number': product.get('rating_number', 0) if product.get('rating_number') else 0,
                        'description': ' '.join(product.get('description', []))[:500] if product.get('description') else '',
                        'features': '|'.join(product.get('features', []))[:500] if product.get('features') else '',
                        'images': '|'.join(product.get('images', [])[:3]) if product.get('images') else '',  # Top 3 image URLs
                    }

                    all_products.append(product_data)
                    products_loaded += 1

                except (json.JSONDecodeError, KeyError) as e:
                    continue

        console.print(f"  ✓ Loaded {products_loaded:,} products\n")

    # Convert to Polars DataFrame
    console.print(f"Creating products DataFrame with {len(all_products):,} total products...")

    df = pl.DataFrame(all_products)

    # Ensure consistent schema
    df = df.with_columns([
        pl.col('article_id').cast(pl.Utf8),
        pl.col('product_code').cast(pl.Utf8),
        pl.col('prod_name').cast(pl.Utf8),
        pl.col('main_category').cast(pl.Utf8),
        pl.col('category').cast(pl.Utf8),
        pl.col('brand').cast(pl.Utf8),
        pl.col('price').cast(pl.Float64),
        pl.col('rating').cast(pl.Float64),
        pl.col('rating_number').cast(pl.Int64),
        pl.col('description').cast(pl.Utf8),
        pl.col('features').cast(pl.Utf8),
        pl.col('images').cast(pl.Utf8),
    ])

    # Save to Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "products_sample.parquet"

    df.write_parquet(output_file)

    console.print(f"✓ Saved {len(df):,} products to {output_file}")


def load_amazon_reviews(data_dir: Path, output_dir: Path, target_customers: int = 100_000) -> None:
    """Load Amazon reviews into transactions table format.

    Args:
        data_dir: Directory containing Amazon JSONL.gz files
        output_dir: Output directory for Parquet files
        target_customers: Target number of customers to sample
    """
    console.print(f"\n[bold blue]Loading Amazon reviews (targeting {target_customers:,} customers)...[/bold blue]\n")

    all_transactions = []
    customer_review_counts = {}

    categories = [
        "All_Beauty",
        "Clothing_Shoes_and_Jewelry",
        "Electronics",
        "Books",
        "Sports_and_Outdoors",
    ]

    # First pass: count reviews per customer
    console.print("[1/2] Counting customer activity...")

    for category in categories:
        review_file = data_dir / f"reviews_sampled_{category}.jsonl.gz"

        if not review_file.exists():
            continue

        with gzip.open(review_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    user_id = review.get('user_id')

                    if user_id:
                        customer_review_counts[user_id] = customer_review_counts.get(user_id, 0) + 1

                except json.JSONDecodeError:
                    continue

    # Sample top N customers by review count
    top_customers = sorted(
        customer_review_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:target_customers]

    sampled_customers = {user_id for user_id, _ in top_customers}

    console.print(f"  Selected {len(sampled_customers):,} customers with most reviews")
    console.print(f"  Avg reviews per customer: {sum(c for _, c in top_customers) / len(top_customers):.1f}\n")

    # Second pass: load reviews for sampled customers
    console.print("[2/2] Loading reviews for sampled customers...")

    transaction_id = 1

    for category in categories:
        console.print(f"  {category}")

        review_file = data_dir / f"reviews_sampled_{category}.jsonl.gz"

        if not review_file.exists():
            console.print(f"    ⚠ Skipping (file not found)")
            continue

        reviews_loaded = 0

        with gzip.open(review_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    user_id = review.get('user_id')

                    if user_id not in sampled_customers:
                        continue

                    # Parse timestamp
                    timestamp = review.get('timestamp')
                    if timestamp:
                        try:
                            transaction_date = datetime.fromtimestamp(timestamp / 1000).date()
                        except:
                            transaction_date = datetime(2023, 1, 1).date()
                    else:
                        transaction_date = datetime(2023, 1, 1).date()

                    transaction_data = {
                        'id': transaction_id,
                        'customer_id': user_id,
                        'article_id': review.get('parent_asin'),
                        'transaction_date': transaction_date,
                        'price': review.get('price', 0.0) if review.get('price') else 0.0,
                        'rating': float(review.get('rating', 0.0)) if review.get('rating') else 0.0,
                    }

                    all_transactions.append(transaction_data)
                    transaction_id += 1
                    reviews_loaded += 1

                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        console.print(f"    ✓ Loaded {reviews_loaded:,} transactions")

    # Convert to Polars DataFrame
    console.print(f"\nCreating transactions DataFrame with {len(all_transactions):,} transactions...")

    df = pl.DataFrame(all_transactions)

    df = df.with_columns([
        pl.col('id').cast(pl.Int64),
        pl.col('customer_id').cast(pl.Utf8),
        pl.col('article_id').cast(pl.Utf8),
        pl.col('transaction_date').cast(pl.Date),
        pl.col('price').cast(pl.Float64),
        pl.col('rating').cast(pl.Float64),
    ])

    # Save to Parquet
    output_file = output_dir / "transactions_sample.parquet"
    df.write_parquet(output_file)

    console.print(f"✓ Saved {len(df):,} transactions to {output_file}")


def load_amazon_customers(transactions_file: Path, output_dir: Path) -> None:
    """Generate customers table from transactions.

    Args:
        transactions_file: Path to transactions Parquet file
        output_dir: Output directory for Parquet files
    """
    console.print("\n[bold blue]Generating customers table...[/bold blue]\n")

    # Load transactions
    df = pl.read_parquet(transactions_file)

    # Extract unique customers
    customers = df.select('customer_id').unique().with_columns([
        pl.lit(None).cast(pl.Utf8).alias('age_bracket'),
        pl.lit(None).cast(pl.Utf8).alias('club_member_status'),
        pl.lit(None).cast(pl.Utf8).alias('fashion_news_frequency'),
    ])

    # Save to Parquet
    output_file = output_dir / "customers_sample.parquet"
    customers.write_parquet(output_file)

    console.print(f"✓ Saved {len(customers):,} customers to {output_file}")


def main():
    console.print("[bold]Amazon Data Loader[/bold]\n")

    data_dir = Path("data/amazon")
    output_dir = Path("data/amazon_sample")

    # Load products
    load_amazon_products(data_dir, output_dir)

    # Load reviews/transactions
    load_amazon_reviews(data_dir, output_dir, target_customers=100_000)

    # Generate customers
    transactions_file = output_dir / "transactions_sample.parquet"
    if transactions_file.exists():
        load_amazon_customers(transactions_file, output_dir)

    console.print("\n[bold green]✓ Amazon data loading complete![/bold green]")
    console.print(f"\nData saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
