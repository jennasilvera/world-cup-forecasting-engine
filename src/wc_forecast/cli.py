from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wc_forecast.data.ingest_results import load_historical_results, save_processed_results
from wc_forecast.features.build_features import save_match_features
from wc_forecast.models.classifier import save_logistic_backtest
from wc_forecast.models.elo import EloModel
from wc_forecast.reporting.prediction_report import save_backtest_report

DEFAULT_PROCESSED_RESULTS_PATH = Path("data/processed/results.csv")
DEFAULT_FEATURES_PATH = Path("data/processed/features.csv")
DEFAULT_ELO_RATINGS_PATH = Path("outputs/elo_ratings.csv")
DEFAULT_ELO_HISTORY_PATH = Path("outputs/elo_history.csv")
DEFAULT_LOGISTIC_PREDICTIONS_PATH = Path("outputs/logistic_backtest_predictions.csv")
DEFAULT_LOGISTIC_METRICS_PATH = Path("outputs/logistic_backtest_metrics.csv")
DEFAULT_BACKTEST_REPORT_PATH = Path("reports/logistic_backtest_report.md")

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


@app.command("build-elo")
def build_elo(
    results_path: Annotated[
        Path,
        typer.Argument(help="Path to processed historical results CSV."),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    ratings_output: Annotated[
        Path,
        typer.Option(
            "--ratings-output",
            help="Path for latest Elo ratings CSV.",
        ),
    ] = DEFAULT_ELO_RATINGS_PATH,
    history_output: Annotated[
        Path,
        typer.Option(
            "--history-output",
            help="Path for match-level Elo history CSV.",
        ),
    ] = DEFAULT_ELO_HISTORY_PATH,
) -> None:
    """Build custom Elo ratings from historical match results."""
    results = load_historical_results(results_path)

    model = EloModel()
    history = model.fit(results)
    ratings = model.ratings_table()

    ratings_output.parent.mkdir(parents=True, exist_ok=True)
    history_output.parent.mkdir(parents=True, exist_ok=True)

    ratings.to_csv(ratings_output, index=False)
    history.to_csv(history_output, index=False)

    table = Table(title="Top Elo Ratings")
    table.add_column("Rank", justify="right")
    table.add_column("Team")
    table.add_column("Elo Rating", justify="right")

    for rank, row in enumerate(ratings.head(10).itertuples(index=False), start=1):
        table.add_row(str(rank), row.team, f"{row.elo_rating:.1f}")

    console.print(table)
    console.print(f"[green]Elo ratings written to:[/green] {ratings_output}")
    console.print(f"[green]Elo history written to:[/green] {history_output}")


@app.command("build-features")
def build_features(
    results_path: Annotated[
        Path,
        typer.Argument(help="Path to processed historical results CSV."),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for model-ready feature table CSV.",
        ),
    ] = DEFAULT_FEATURES_PATH,
) -> None:
    """Build a model-ready pre-match feature table."""
    destination = save_match_features(results_path=results_path, output_path=output_path)
    console.print(f"[green]Feature table written to:[/green] {destination}")


@app.command("backtest-logistic")
def backtest_logistic(
    features_path: Annotated[
        Path,
        typer.Argument(help="Path to model-ready feature table CSV."),
    ] = DEFAULT_FEATURES_PATH,
    predictions_output: Annotated[
        Path,
        typer.Option(
            "--predictions-output",
            help="Path for match-level backtest predictions CSV.",
        ),
    ] = DEFAULT_LOGISTIC_PREDICTIONS_PATH,
    metrics_output: Annotated[
        Path,
        typer.Option(
            "--metrics-output",
            help="Path for backtest metrics CSV.",
        ),
    ] = DEFAULT_LOGISTIC_METRICS_PATH,
    test_fraction: Annotated[
        float,
        typer.Option(
            "--test-fraction",
            help="Fraction of latest matches reserved for chronological test set.",
        ),
    ] = 0.30,
) -> None:
    """Run a chronological logistic-regression backtest."""
    result = save_logistic_backtest(
        features_path=features_path,
        predictions_output_path=predictions_output,
        metrics_output_path=metrics_output,
        test_fraction=test_fraction,
    )

    table = Table(title="Logistic Regression Backtest Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for row in result.metrics.itertuples(index=False):
        table.add_row(row.metric, f"{row.value:.4f}")

    console.print(table)
    console.print(f"[green]Predictions written to:[/green] {predictions_output}")
    console.print(f"[green]Metrics written to:[/green] {metrics_output}")


@app.command("report-backtest")
def report_backtest(
    predictions_path: Annotated[
        Path,
        typer.Argument(help="Path to match-level backtest predictions CSV."),
    ] = DEFAULT_LOGISTIC_PREDICTIONS_PATH,
    metrics_path: Annotated[
        Path,
        typer.Option(
            "--metrics-path",
            help="Path to backtest metrics CSV.",
        ),
    ] = DEFAULT_LOGISTIC_METRICS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for generated Markdown report.",
        ),
    ] = DEFAULT_BACKTEST_REPORT_PATH,
) -> None:
    """Generate a recruiter-facing Markdown backtest report."""
    destination = save_backtest_report(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        output_path=output_path,
    )

    console.print(f"[green]Backtest report written to:[/green] {destination}")


if __name__ == "__main__":
    app()
