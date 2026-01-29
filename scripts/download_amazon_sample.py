"""Download and sample Amazon product data for diverse e-commerce demo."""

import gzip
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import polars as pl
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Category configurations
CATEGORIES = {
    "All_Beauty": {
        "meta_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz",
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz",
        "target_products": 30000,
        "needs_images": False,  # Text embeddings
    },
    "Clothing_Shoes_and_Jewelry": {
        "meta_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz",
        "target_products": 30000,
        "needs_images": True,  # Visual similarity
    },
    "Electronics": {
        "meta_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz",
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
        "target_products": 40000,
        "needs_images": False,  # Text embeddings
    },
    "Books": {
        "meta_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz",
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz",
        "target_products": 30000,
        "needs_images": False,  # Genre/author based
    },
    "Sports_and_Outdoors": {
        "meta_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz",
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz",
        "target_products": 20000,
        "needs_images": False,  # Category + collaborative
    },
}


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    console.print(f"Downloading {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (10 * 1024 * 1024) == 0:  # Every 10MB
                    console.print(f"  Downloaded: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB")

    console.print(f"  ✓ Saved to {output_path}")


def sample_products_from_metadata(
    meta_path: Path,
    target_count: int,
    needs_images: bool,
    min_reviews: int = 5,
) -> list[str]:
    """Sample products from metadata file.

    Prioritizes:
    - Products with images (if needs_images=True)
    - Products with more reviews
    - Random sampling from qualified products
    """
    console.print(f"  Sampling {target_count:,} products from {meta_path.name}...")

    products_with_reviews = []

    with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0 and line_num > 0:
                console.print(f"    Processed {line_num:,} products...")

            try:
                product = json.loads(line)
                parent_asin = product.get('parent_asin')

                # Skip if missing parent_asin (primary product ID)
                if not parent_asin:
                    continue

                # Check image requirement
                if needs_images:
                    images = product.get('images', [])
                    if not images or len(images) == 0:
                        continue

                # Estimate review count (if available)
                rating_number = product.get('rating_number', 0)

                if rating_number >= min_reviews:
                    products_with_reviews.append({
                        'asin': parent_asin,
                        'reviews': rating_number,
                        'title': product.get('title', ''),
                        'has_images': bool(product.get('images')),
                    })

            except json.JSONDecodeError:
                continue

    console.print(f"    Found {len(products_with_reviews):,} qualified products")

    # Sample products (weighted by review count)
    if len(products_with_reviews) <= target_count:
        sampled = products_with_reviews
    else:
        # Sort by review count and take top N with some randomization
        products_with_reviews.sort(key=lambda x: x['reviews'], reverse=True)

        # Take top 50% of target as high-review products
        top_portion = products_with_reviews[:target_count // 2]

        # Randomly sample remaining from rest
        remaining_portion = random.sample(
            products_with_reviews[target_count // 2:],
            min(target_count - len(top_portion), len(products_with_reviews) - target_count // 2)
        )

        sampled = top_portion + remaining_portion

    console.print(f"    ✓ Sampled {len(sampled):,} products")

    return [p['asin'] for p in sampled]


def filter_reviews_by_products(
    review_path: Path,
    product_asins: set[str],
    output_path: Path,
) -> int:
    """Filter reviews to only include sampled products."""
    console.print(f"  Filtering reviews for {len(product_asins):,} products...")

    reviews_kept = 0

    with gzip.open(review_path, 'rt', encoding='utf-8') as fin:
        with gzip.open(output_path, 'wt', encoding='utf-8') as fout:
            for line_num, line in enumerate(fin):
                if line_num % 1000000 == 0 and line_num > 0:
                    console.print(f"    Processed {line_num:,} reviews, kept {reviews_kept:,}...")

                try:
                    review = json.loads(line)
                    parent_asin = review.get('parent_asin')

                    if parent_asin in product_asins:
                        fout.write(line)
                        reviews_kept += 1

                except json.JSONDecodeError:
                    continue

    console.print(f"    ✓ Kept {reviews_kept:,} reviews")
    return reviews_kept


def main():
    console.print("[bold blue]Amazon E-Commerce Sample Download[/bold blue]\n")

    data_dir = Path("data/amazon")
    data_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    all_sampled_products = {}

    # Step 1: Download metadata and sample products
    console.print("[bold]Step 1: Downloading metadata and sampling products[/bold]\n")

    for category_name, config in CATEGORIES.items():
        console.print(f"[bold cyan]{category_name}[/bold cyan]")

        meta_path = data_dir / f"meta_{category_name}.jsonl.gz"

        # Download metadata if not exists
        if not meta_path.exists():
            download_file(config['meta_url'], meta_path)
        else:
            console.print(f"  Using cached {meta_path.name}")

        # Sample products
        sampled_asins = sample_products_from_metadata(
            meta_path=meta_path,
            target_count=config['target_products'],
            needs_images=config['needs_images'],
        )

        all_sampled_products[category_name] = sampled_asins

        # Save sampled product list
        sample_file = data_dir / f"sampled_products_{category_name}.txt"
        with open(sample_file, 'w') as f:
            f.write('\n'.join(sampled_asins))

        console.print(f"  ✓ Saved product list to {sample_file.name}\n")

    # Step 2: Download and filter reviews
    console.print("\n[bold]Step 2: Downloading and filtering reviews[/bold]\n")

    total_reviews = 0

    for category_name, config in CATEGORIES.items():
        console.print(f"[bold cyan]{category_name}[/bold cyan]")

        review_path = data_dir / f"reviews_{category_name}.jsonl.gz"
        filtered_review_path = data_dir / f"reviews_sampled_{category_name}.jsonl.gz"

        # Download reviews if not exists
        if not review_path.exists():
            download_file(config['review_url'], review_path)
        else:
            console.print(f"  Using cached {review_path.name}")

        # Filter reviews
        product_asins = set(all_sampled_products[category_name])
        reviews_count = filter_reviews_by_products(
            review_path=review_path,
            product_asins=product_asins,
            output_path=filtered_review_path,
        )

        total_reviews += reviews_count
        console.print()

    # Summary
    console.print("\n[bold green]✓ Download and sampling complete![/bold green]\n")
    console.print("Summary:")
    for category_name, asins in all_sampled_products.items():
        console.print(f"  {category_name}: {len(asins):,} products")
    console.print(f"\n  Total reviews: {total_reviews:,}")
    console.print(f"\n  Data saved to: {data_dir.absolute()}")


if __name__ == "__main__":
    main()
