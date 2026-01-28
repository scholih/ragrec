"""CLI application for RagRec."""

import typer
from rich.console import Console

app = typer.Typer(help="RagRec: Multi-modal e-retail recommendation system")
console = Console()


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
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


if __name__ == "__main__":
    app()
