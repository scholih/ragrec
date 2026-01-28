#!/usr/bin/env python3
"""Create sample dataset from H&M data for development and testing.

This script uses Polars for fast data processing to create a representative
sample of the full H&M dataset:
- 1000 products (articles) with available images
- 500 customers who purchased these products
- 5000 transactions linking them
- 100 sample product images

Output files are saved as Parquet in data/sample/.
"""

import shutil
from pathlib import Path

import polars as pl


def main() -> None:
    """Create sample dataset from full H&M data."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  H&M Sample Data Creation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    # Paths
    data_dir = Path("data/hm")
    sample_dir = Path("data/sample")
    sample_images_dir = sample_dir / "images"

    # Create output directories
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load articles and filter for those with images
    print("Step 1: Loading articles with images...")
    articles = pl.read_csv(data_dir / "articles.csv")
    print(f"  Total articles: {len(articles):,}")

    # Get list of available image files
    images_dir = data_dir / "images"
    available_images = set()
    for img_path in images_dir.rglob("*.jpg"):
        article_id = img_path.stem  # Filename without extension (e.g., "0108775015")
        # Remove leading zero to match integer IDs in CSV
        article_id_int = int(article_id)
        available_images.add(article_id_int)

    print(f"  Available images: {len(available_images):,}")

    # Filter articles with images and sample 1000
    articles_with_images = articles.filter(
        pl.col("article_id").is_in(list(available_images))
    )
    print(f"  Articles with images: {len(articles_with_images):,}")

    # Sample 1000 articles (or all if less than 1000)
    sample_size = min(1000, len(articles_with_images))
    articles_sample = articles_with_images.sample(n=sample_size, seed=42)
    print(f"  Sampled articles: {len(articles_sample):,}")

    # Get article IDs for filtering
    sampled_article_ids = set(articles_sample["article_id"].to_list())

    # Step 2: Load transactions and filter for sampled articles
    print()
    print("Step 2: Loading transactions for sampled articles...")
    transactions = pl.read_csv(
        data_dir / "transactions_train.csv",
        columns=["t_dat", "customer_id", "article_id", "price"],
    )
    print(f"  Total transactions: {len(transactions):,}")

    # Filter transactions for sampled articles
    transactions_filtered = transactions.filter(
        pl.col("article_id").is_in(list(sampled_article_ids))
    )
    print(f"  Transactions for sampled articles: {len(transactions_filtered):,}")

    # Sample 5000 transactions (or all if less)
    trans_sample_size = min(5000, len(transactions_filtered))
    transactions_sample = transactions_filtered.sample(n=trans_sample_size, seed=42)
    print(f"  Sampled transactions: {len(transactions_sample):,}")

    # Get customer IDs from sampled transactions
    sampled_customer_ids = set(transactions_sample["customer_id"].unique().to_list())
    print(f"  Unique customers in transactions: {len(sampled_customer_ids):,}")

    # Step 3: Load customers and filter for those in sampled transactions
    print()
    print("Step 3: Loading customers...")
    customers = pl.read_csv(data_dir / "customers.csv")
    print(f"  Total customers: {len(customers):,}")

    customers_sample = customers.filter(
        pl.col("customer_id").is_in(list(sampled_customer_ids))
    )
    print(f"  Sampled customers: {len(customers_sample):,}")

    # Step 4: Save sample data as Parquet
    print()
    print("Step 4: Saving sample data as Parquet...")

    articles_output = sample_dir / "articles_sample.parquet"
    customers_output = sample_dir / "customers_sample.parquet"
    transactions_output = sample_dir / "transactions_sample.parquet"

    articles_sample.write_parquet(articles_output)
    print(f"  ✓ Saved {articles_output} ({len(articles_sample):,} rows)")

    customers_sample.write_parquet(customers_output)
    print(f"  ✓ Saved {customers_output} ({len(customers_sample):,} rows)")

    transactions_sample.write_parquet(transactions_output)
    print(f"  ✓ Saved {transactions_output} ({len(transactions_sample):,} rows)")

    # Step 5: Copy 100 sample images
    print()
    print("Step 5: Copying sample images...")

    # Get first 100 article IDs
    image_article_ids = list(sampled_article_ids)[:100]
    copied_count = 0

    for article_id in image_article_ids:
        # Convert int to string with leading zero (10 digits total)
        article_id_str = str(article_id).zfill(10)
        # Find the image file (it's in a subdirectory based on first 3 digits)
        prefix = article_id_str[:3]
        source_path = images_dir / prefix / f"{article_id_str}.jpg"

        if source_path.exists():
            dest_path = sample_images_dir / f"{article_id_str}.jpg"
            shutil.copy2(source_path, dest_path)
            copied_count += 1

    print(f"  ✓ Copied {copied_count} images to {sample_images_dir}")

    # Step 6: Create README
    print()
    print("Step 6: Creating README...")

    readme_content = f"""# H&M Sample Dataset

Sample dataset created from H&M Fashion Recommendations dataset for development and testing.

## Contents

- **articles_sample.parquet** - {len(articles_sample):,} products with images
- **customers_sample.parquet** - {len(customers_sample):,} customers who purchased these products
- **transactions_sample.parquet** - {len(transactions_sample):,} purchase transactions
- **images/** - {copied_count} product images

## Statistics

| File | Rows | Columns |
|------|------|---------|
| articles_sample.parquet | {len(articles_sample):,} | {len(articles_sample.columns)} |
| customers_sample.parquet | {len(customers_sample):,} | {len(customers_sample.columns)} |
| transactions_sample.parquet | {len(transactions_sample):,} | {len(transactions_sample.columns)} |

## Schema

### Articles
{articles_sample.schema}

### Customers
{customers_sample.schema}

### Transactions
{transactions_sample.schema}

## Usage

```python
import polars as pl

# Load sample data
articles = pl.read_parquet("data/sample/articles_sample.parquet")
customers = pl.read_parquet("data/sample/customers_sample.parquet")
transactions = pl.read_parquet("data/sample/transactions_sample.parquet")
```

## Source

Full dataset: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

Sample created: 2026-01-28 (via create_sample_data.py)
Sampling seed: 42 (reproducible)
"""

    readme_path = sample_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"  ✓ Created {readme_path}")

    # Summary
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  ✅ Sample Data Creation Complete!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print(f"Sample data saved to: {sample_dir.absolute()}")
    print()
    print("Files created:")
    print(f"  • articles_sample.parquet ({len(articles_sample):,} products)")
    print(f"  • customers_sample.parquet ({len(customers_sample):,} customers)")
    print(f"  • transactions_sample.parquet ({len(transactions_sample):,} transactions)")
    print(f"  • images/ ({copied_count} JPG files)")
    print(f"  • README.md")
    print()


if __name__ == "__main__":
    main()
