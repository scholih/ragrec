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


if __name__ == "__main__":
    app()
