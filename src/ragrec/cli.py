"""CLI application for RagRec."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="RagRec: Multi-modal e-retail recommendation system")
console = Console()


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 9010,
    reload: bool = False,
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[bold green]Starting RagRec API on {host}:{port}[/bold green]")
    uvicorn.run(
        "ragrec.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def ui() -> None:
    """Start the Streamlit UI (not yet implemented)."""
    console.print("[bold yellow]Streamlit UI not yet implemented[/bold yellow]")
    console.print("This will be added in Phase 3")


@app.command()
def version() -> None:
    """Show version information."""
    from ragrec import __version__

    console.print(f"[bold]RagRec[/bold] version [green]{__version__}[/green]")


@app.command()
def load_data(
    data_path: Path = typer.Argument(
        ...,
        help="Path to directory containing Parquet files (e.g., data/sample)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear existing data before loading",
    ),
) -> None:
    """Load H&M data into PostgreSQL database."""
    from ragrec.etl import load_hm_data

    try:
        asyncio.run(load_hm_data(data_path, clear_first=clear))
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def generate_embeddings(
    images_dir: Path = typer.Argument(
        ...,
        help="Path to directory containing product images (e.g., data/sample/images)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    model: str = typer.Option(
        "google/siglip-base-patch16-224",
        "--model",
        help="SigLIP model identifier from HuggingFace",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Number of images to process per batch",
    ),
) -> None:
    """Generate SigLIP embeddings for product images."""
    from ragrec.embeddings import generate_product_embeddings

    try:
        asyncio.run(
            generate_product_embeddings(
                images_dir,
                model_name=model,
                batch_size=batch_size,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def create_index(
    ef_construction: int = typer.Option(
        128,
        "--ef-construction",
        help="HNSW ef_construction parameter (higher = better recall)",
    ),
    m: int = typer.Option(
        16,
        "--m",
        help="HNSW m parameter (connections per layer)",
    ),
) -> None:
    """Create HNSW index on embedding column for fast similarity search."""
    from ragrec.vectorstore import PgVectorStore

    async def _create_index() -> None:
        console.print("[bold blue]Creating HNSW index...[/bold blue]")
        console.print(f"  Parameters: ef_construction={ef_construction}, m={m}")

        async with PgVectorStore() as store:
            await store.create_hnsw_index(ef_construction=ef_construction, m=m)
            stats = await store.get_index_stats()

        console.print("[bold green]✓ HNSW index created successfully![/bold green]")
        console.print(f"  Index size: {stats['size']}")
        console.print(f"  Total embeddings: {stats['total_embeddings']:,}")

    try:
        asyncio.run(_create_index())
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    image_path: Path = typer.Argument(
        ...,
        help="Path to query image",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="Number of similar products to return",
    ),
    category: str = typer.Option(
        None,
        "--category",
        help="Filter by product category (e.g., 'Trousers', 'Dress')",
    ),
) -> None:
    """Search for visually similar products using an image."""
    from ragrec.recommender import VisualRecommender

    async def _search() -> None:
        # Load recommender
        console.print("[bold blue]Loading visual recommender...[/bold blue]")

        # Search for similar products
        console.print(f"[bold blue]Searching for top {top_k} similar products...[/bold blue]")
        image_bytes = image_path.read_bytes()

        async with VisualRecommender() as recommender:
            results = await recommender.find_similar(
                image_bytes=image_bytes,
                top_k=top_k,
                category_filter=category,
            )

        # Display results
        console.print("\n[bold green]Search Results:[/bold green]")
        if not results.is_empty():
            console.print(results)
        else:
            console.print("[yellow]No results found matching criteria[/yellow]")

    try:
        asyncio.run(_search())
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def similar(
    product_id: int = typer.Argument(
        ...,
        help="Article ID of the reference product",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="Number of similar products to return",
    ),
    category: str = typer.Option(
        None,
        "--category",
        help="Filter by product category",
    ),
) -> None:
    """Find products similar to an existing product by ID."""
    from ragrec.recommender import VisualRecommender

    async def _similar() -> None:
        console.print(f"[bold blue]Finding products similar to {product_id}...[/bold blue]")

        async with VisualRecommender() as recommender:
            results = await recommender.find_similar_by_id(
                product_id=product_id,
                top_k=top_k,
                category_filter=category,
            )

        # Display results
        console.print("\n[bold green]Similar Products:[/bold green]")
        if not results.is_empty():
            console.print(results)
        else:
            console.print("[yellow]No similar products found or product ID not in database[/yellow]")

    try:
        asyncio.run(_similar())
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def load_graph(
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear existing graph data before loading",
    ),
    top_k_similar: int = typer.Option(
        5,
        "--top-k-similar",
        help="Number of similar products per product to create SIMILAR_TO edges",
    ),
) -> None:
    """Load data from PostgreSQL into Neo4j graph database."""
    from ragrec.etl.graph_loader import load_graph_from_postgres

    try:
        asyncio.run(load_graph_from_postgres(clear_first=clear, top_k_similar=top_k_similar))
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
