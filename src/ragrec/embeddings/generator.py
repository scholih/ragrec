"""Generate and store embeddings for products."""

import asyncio
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ragrec.embeddings.siglip import SigLIPEmbedder
from ragrec.etl.config import ETLConfig

console = Console()


class EmbeddingGenerator:
    """Generate SigLIP embeddings for product images."""

    def __init__(
        self,
        embedder: SigLIPEmbedder,
        config: ETLConfig | None = None,
    ) -> None:
        """Initialize embedding generator.

        Args:
            embedder: SigLIP embedder instance
            config: ETL configuration
        """
        self.embedder = embedder
        self.config = config or ETLConfig()
        self.pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "EmbeddingGenerator":
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

    async def get_products_without_embeddings(self) -> list[dict[str, Any]]:
        """Get products that don't have embeddings yet."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p.article_id
                FROM products p
                LEFT JOIN product_embeddings pe ON p.article_id = pe.article_id
                WHERE pe.article_id IS NULL
                ORDER BY p.article_id
                """
            )

        return [dict(row) for row in rows]

    async def store_embedding(
        self, article_id: int, embedding: np.ndarray
    ) -> None:
        """Store embedding for a product."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        # Convert numpy array to string format for pgvector: "[1.0, 2.0, 3.0]"
        embedding_str = "[" + ",".join(str(x) for x in embedding.tolist()) + "]"

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO product_embeddings (article_id, embedding, model_version)
                VALUES ($1, $2::vector, $3)
                ON CONFLICT (article_id) DO UPDATE
                SET embedding = $2::vector, model_version = $3, created_at = CURRENT_TIMESTAMP
                """,
                article_id,
                embedding_str,
                self.embedder.model_name,
            )

    async def generate_embeddings(
        self, images_dir: Path, batch_size: int = 32
    ) -> dict[str, int]:
        """Generate embeddings for all products.

        Args:
            images_dir: Directory containing product images
            batch_size: Number of images to process in each batch

        Returns:
            Statistics dictionary with counts
        """
        console.print("\n[bold blue]Fetching products without embeddings...[/bold blue]")

        products = await self.get_products_without_embeddings()
        total_products = len(products)

        if total_products == 0:
            console.print("  ✓ All products already have embeddings")
            return {"total": 0, "success": 0, "skipped": 0, "failed": 0}

        console.print(f"  Found {total_products:,} products to process")

        stats = {"total": total_products, "success": 0, "skipped": 0, "failed": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generating embeddings...", total=total_products
            )

            # Process in batches
            for i in range(0, total_products, batch_size):
                batch = products[i : i + batch_size]

                # Load images for this batch
                batch_images = []
                batch_article_ids = []

                for product in batch:
                    article_id = product["article_id"]
                    # Convert article_id to image filename (10 digits with leading zeros)
                    article_id_str = str(article_id).zfill(10)
                    prefix = article_id_str[:3]

                    # Check both data/sample/images and data/hm/images
                    image_path = images_dir / f"{article_id_str}.jpg"
                    if not image_path.exists():
                        # Try data/hm/images with subdirectory structure
                        image_path = (
                            images_dir.parent.parent / "hm" / "images" / prefix / f"{article_id_str}.jpg"
                        )

                    if image_path.exists():
                        try:
                            image_bytes = image_path.read_bytes()
                            batch_images.append(image_bytes)
                            batch_article_ids.append(article_id)
                        except Exception as e:
                            console.print(
                                f"  [yellow]Warning: Failed to load {image_path}: {e}[/yellow]"
                            )
                            stats["failed"] += 1
                    else:
                        stats["skipped"] += 1

                # Generate embeddings for batch
                if batch_images:
                    try:
                        embeddings = self.embedder.batch_encode_images(
                            batch_images, batch_size=len(batch_images)
                        )

                        # Store embeddings
                        for article_id, embedding in zip(batch_article_ids, embeddings):
                            await self.store_embedding(article_id, embedding)
                            stats["success"] += 1

                    except Exception as e:
                        console.print(
                            f"  [red]Error processing batch: {e}[/red]"
                        )
                        stats["failed"] += len(batch_images)

                progress.update(task, advance=len(batch))

        return stats


async def generate_product_embeddings(
    images_dir: Path,
    model_name: str = "google/siglip-base-patch16-224",
    batch_size: int = 32,
) -> None:
    """Generate SigLIP embeddings for all products.

    Args:
        images_dir: Directory containing product images
        model_name: SigLIP model identifier
        batch_size: Batch size for processing
    """
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("  SigLIP Embedding Generation")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"\nImages directory: {images_dir.absolute()}")
    console.print(f"Model: {model_name}")
    console.print(f"Batch size: {batch_size}")

    # Initialize SigLIP embedder
    console.print("\n[bold blue]Loading SigLIP model...[/bold blue]")
    embedder = SigLIPEmbedder(model_name=model_name)
    console.print(f"  ✓ Model loaded on device: {embedder.device}")
    console.print(f"  ✓ Embedding dimension: {embedder.embedding_dim}")

    # Generate embeddings
    async with EmbeddingGenerator(embedder) as generator:
        stats = await generator.generate_embeddings(images_dir, batch_size=batch_size)

    # Summary
    console.print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("  ✅ Embedding Generation Complete!")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"\nProcessed: {stats['total']:,} products")
    console.print(f"  • Success: {stats['success']:,}")
    console.print(f"  • Skipped (no image): {stats['skipped']:,}")
    console.print(f"  • Failed: {stats['failed']:,}")
    console.print()


if __name__ == "__main__":
    # For testing
    asyncio.run(generate_product_embeddings(Path("data/sample/images")))
