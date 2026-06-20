from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from wc_forecast.data.ingest_results import save_processed_results

DEFAULT_PROCESSED_RESULTS_PATH = Path("data/processed/results.csv")

app = typer.Typer(
    help="World Cup Match Forecasting Engine CLI",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Quant-style probabilistic football forecasting engine."""
    return None


@app.command()
def health() -> None:
    """Check that the project CLI is installed and runnable."""
    console.print("[green]World Cup Forecasting Engine is ready.[/green]")


@app.command("ingest-results")
def ingest_results(
    input_path: Annotated[
        Path,
        typer.Argument(help="Path to raw historical results CSV."),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for cleaned processed results CSV.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
) -> None:
    """Validate and process historical international football results."""
    destination = save_processed_results(input_path=input_path, output_path=output_path)
    console.print(f"[green]Processed results written to:[/green] {destination}")


if __name__ == "__main__":
    app()
