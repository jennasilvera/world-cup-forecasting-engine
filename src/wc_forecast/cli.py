from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="World Cup Match Forecasting Engine CLI")
console = Console()


@app.command()
def health() -> None:
    """Check that the project CLI is installed and runnable."""
    console.print("[green]World Cup Forecasting Engine is ready.[/green]")


if __name__ == "__main__":
    app()
