from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from wc_forecast.data.ingest_fixtures import save_world_cup_fixtures
from wc_forecast.data.ingest_results import load_historical_results, save_processed_results
from wc_forecast.data_sources.international_results import save_normalized_international_results
from wc_forecast.features.build_features import save_match_features
from wc_forecast.forecasting.fixture_forecast import (
    save_fixture_forecasts_from_results,
    save_upcoming_fixture_forecasts_from_results,
)
from wc_forecast.ledger.prediction_ledger import (
    append_candidate_edges_to_prediction_ledger,
    save_market_prediction_to_ledger,
    settle_prediction_ledger_from_results,
    settle_prediction_ledger_row,
)
from wc_forecast.models.batch_market import save_market_odds_slate_evaluation
from wc_forecast.models.classifier import (
    DEFAULT_LOGISTIC_C,
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    save_logistic_backtest,
)
from wc_forecast.models.elo import EloModel
from wc_forecast.models.market import (
    calculate_market_edge,
    calculate_market_probabilities,
)
from wc_forecast.models.poisson import PoissonGoalsModel, save_poisson_prediction
from wc_forecast.reporting.group_stage_report import save_group_stage_report
from wc_forecast.reporting.ledger_report import save_prediction_ledger_report
from wc_forecast.reporting.match_report import (
    generate_match_prediction,
    save_match_prediction_report,
)
from wc_forecast.reporting.prediction_report import save_backtest_report
from wc_forecast.reports.artifact_index import save_artifact_index
from wc_forecast.reports.upcoming_forecast_report import save_upcoming_forecast_report
from wc_forecast.simulation.group_stage import save_group_stage_simulation
from wc_forecast.strategy.policy import StrategyPolicy, save_strategy_policy_output
from wc_forecast.strategy.staking import StakeSizingPolicy, save_stake_sizing_output
from wc_forecast.validation.feature_ablation import (
    DEFAULT_FEATURE_SET_NAMES,
    save_feature_ablation,
)
from wc_forecast.validation.logistic_tuning import (
    DEFAULT_HALF_LIFE_DAYS_GRID,
    DEFAULT_LOGISTIC_C_GRID,
    save_logistic_tuning,
)
from wc_forecast.validation.rolling_backtest import (
    DEFAULT_MODEL_TYPES,
    DEFAULT_ROLLING_CUTOFF_DATES,
    save_rolling_backtest,
    summarize_rolling_backtest,
)

DEFAULT_PROCESSED_RESULTS_PATH = Path("data/processed/results.csv")
DEFAULT_REAL_RESULTS_PATH = Path("data/processed/real_international_results.csv")
DEFAULT_FEATURES_PATH = Path("data/processed/features.csv")
DEFAULT_ELO_RATINGS_PATH = Path("outputs/elo_ratings.csv")
DEFAULT_ELO_HISTORY_PATH = Path("outputs/elo_history.csv")
DEFAULT_LOGISTIC_PREDICTIONS_PATH = Path("outputs/logistic_backtest_predictions.csv")
DEFAULT_LOGISTIC_METRICS_PATH = Path("outputs/logistic_backtest_metrics.csv")
DEFAULT_BACKTEST_REPORT_PATH = Path("reports/logistic_backtest_report.md")
DEFAULT_POISSON_PREDICTION_PATH = Path("outputs/poisson_prediction.csv")
DEFAULT_MATCH_PREDICTION_PATH = Path("outputs/match_prediction.csv")
DEFAULT_MATCH_REPORT_PATH = Path("reports/match_prediction_report.md")
DEFAULT_GROUP_FIXTURES_PATH = Path("data/sample/group_stage_fixtures_sample.csv")
DEFAULT_WORLD_CUP_FIXTURES_PATH = Path("data/sample/world_cup_2026_fixtures_sample.csv")
DEFAULT_WORLD_CUP_FORECASTS_PATH = Path("outputs/world_cup_2026_forecasts.csv")
DEFAULT_GROUP_SIMULATION_PATH = Path("outputs/group_stage_simulation.csv")
DEFAULT_GROUP_SIMULATION_REPORT_PATH = Path("reports/group_stage_simulation_report.md")
DEFAULT_MARKET_EDGE_PATH = Path("outputs/market_edge.csv")
DEFAULT_MARKET_ODDS_PATH = Path("data/sample/market_odds_sample.csv")
DEFAULT_BATCH_MARKET_EDGE_PATH = Path("outputs/batch_market_edges.csv")
DEFAULT_STRATEGY_POLICY_PATH = Path("outputs/strategy_policy_edges.csv")
DEFAULT_STAKE_SIZING_PATH = Path("outputs/stake_sizing_edges.csv")
DEFAULT_SETTLEMENT_RESULTS_PATH = Path("data/sample/settlement_results_sample.csv")
DEFAULT_PREDICTION_LEDGER_PATH = Path("outputs/prediction_ledger.csv")
DEFAULT_PREDICTION_LEDGER_REPORT_PATH = Path("outputs/prediction_ledger_report.md")

DEFAULT_WORLD_CUP_2026_RAW_FIXTURES_PATH = Path("data/raw/world_cup_2026_fixtures.csv")
DEFAULT_WORLD_CUP_2026_FIXTURES_PATH = Path("data/processed/world_cup_2026_fixtures.csv")
DEFAULT_UPCOMING_FORECAST_RESULTS_PATH = Path("data/processed/results.csv")
DEFAULT_WORLD_CUP_2026_UPCOMING_FORECASTS_PATH = Path(
    "outputs/world_cup_2026_upcoming_forecasts.csv"
)
DEFAULT_WORLD_CUP_2026_UPCOMING_REPORT_PATH = Path(
    "outputs/world_cup_2026_upcoming_forecast_report.md"
)
DEFAULT_ARTIFACT_INDEX_PATH = Path("outputs/forecast_artifact_index.csv")

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



DEFAULT_ROLLING_BACKTEST_METRICS_PATH = Path("outputs/rolling_backtest_metrics.csv")
DEFAULT_LOGISTIC_TUNING_PATH = Path("outputs/logistic_tuning_results.csv")
DEFAULT_FEATURE_ABLATION_PATH = Path("outputs/feature_ablation_results.csv")



@app.command("ablate-features")
def ablate_features_command(
    features_path: Annotated[
        Path,
        typer.Argument(help="Path to model-ready feature table CSV."),
    ] = DEFAULT_FEATURES_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for feature ablation results CSV.",
        ),
    ] = DEFAULT_FEATURE_ABLATION_PATH,
    feature_sets_csv: Annotated[
        str,
        typer.Option(
            "--feature-sets",
            help="Comma-separated feature-set names to compare.",
        ),
    ] = ",".join(DEFAULT_FEATURE_SET_NAMES),
    cutoff_dates_csv: Annotated[
        str,
        typer.Option(
            "--cutoff-dates",
            help="Comma-separated historical cutoff dates.",
        ),
    ] = ",".join(DEFAULT_ROLLING_CUTOFF_DATES),
    evaluation_window_days: Annotated[
        int,
        typer.Option(
            "--evaluation-window-days",
            help="Number of days after each cutoff to evaluate.",
        ),
    ] = 365,
    sample_weight_half_life_days: Annotated[
        float | None,
        typer.Option(
            "--sample-weight-half-life-days",
            help="Optional recency half-life in days for training weights.",
        ),
    ] = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    logistic_c: Annotated[
        float,
        typer.Option(
            "--logistic-c",
            help="Inverse regularization strength for logistic model.",
        ),
    ] = DEFAULT_LOGISTIC_C,
) -> None:
    """Run rolling feature ablation validation."""

    feature_set_names = [
        feature_set.strip()
        for feature_set in feature_sets_csv.split(",")
        if feature_set.strip()
    ]
    cutoff_dates = [
        cutoff.strip()
        for cutoff in cutoff_dates_csv.split(",")
        if cutoff.strip()
    ]

    result = save_feature_ablation(
        features_path=features_path,
        output_path=output_path,
        feature_set_names=feature_set_names,
        cutoff_dates=cutoff_dates,
        evaluation_window_days=evaluation_window_days,
        sample_weight_half_life_days=sample_weight_half_life_days,
        logistic_c=logistic_c,
    )

    table = Table(title="Feature Ablation Validation")
    table.add_column("Feature Set")
    table.add_column("Features", justify="right")
    table.add_column("Folds", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Log Loss", justify="right")
    table.add_column("Brier", justify="right")

    for row in result.itertuples(index=False):
        table.add_row(
            str(row.feature_set),
            str(row.active_feature_count),
            str(row.folds),
            f"{row.mean_accuracy:.4f}",
            f"{row.mean_log_loss:.4f}",
            f"{row.mean_brier:.4f}",
        )

    console.print(table)
    console.print(f"Feature ablation results written to: {output_path}")


@app.command("tune-logistic")
def tune_logistic_command(
    features_path: Annotated[
        Path,
        typer.Argument(help="Path to model-ready feature table CSV."),
    ] = DEFAULT_FEATURES_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for logistic tuning results CSV.",
        ),
    ] = DEFAULT_LOGISTIC_TUNING_PATH,
    half_life_days_csv: Annotated[
        str,
        typer.Option(
            "--half-life-days",
            help="Comma-separated recency half-life values to test.",
        ),
    ] = ",".join(str(int(value)) for value in DEFAULT_HALF_LIFE_DAYS_GRID),
    logistic_c_csv: Annotated[
        str,
        typer.Option(
            "--logistic-c",
            help="Comma-separated logistic C values to test.",
        ),
    ] = ",".join(str(value) for value in DEFAULT_LOGISTIC_C_GRID),
    cutoff_dates_csv: Annotated[
        str,
        typer.Option(
            "--cutoff-dates",
            help="Comma-separated historical cutoff dates.",
        ),
    ] = ",".join(DEFAULT_ROLLING_CUTOFF_DATES),
    evaluation_window_days: Annotated[
        int,
        typer.Option(
            "--evaluation-window-days",
            help="Number of days after each cutoff to evaluate.",
        ),
    ] = 365,
) -> None:
    """Tune logistic model hyperparameters using rolling validation."""

    half_life_days_grid = [
        float(value.strip())
        for value in half_life_days_csv.split(",")
        if value.strip()
    ]
    logistic_c_grid = [
        float(value.strip())
        for value in logistic_c_csv.split(",")
        if value.strip()
    ]
    cutoff_dates = [
        cutoff.strip()
        for cutoff in cutoff_dates_csv.split(",")
        if cutoff.strip()
    ]

    result = save_logistic_tuning(
        features_path=features_path,
        output_path=output_path,
        half_life_days_grid=half_life_days_grid,
        logistic_c_grid=logistic_c_grid,
        cutoff_dates=cutoff_dates,
        evaluation_window_days=evaluation_window_days,
    )

    table = Table(title="Logistic Hyperparameter Tuning")
    table.add_column("Half-Life Days", justify="right")
    table.add_column("C", justify="right")
    table.add_column("Folds", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Log Loss", justify="right")
    table.add_column("Brier", justify="right")

    for row in result.head(10).itertuples(index=False):
        table.add_row(
            f"{row.sample_weight_half_life_days:.0f}",
            f"{row.logistic_c:.2f}",
            str(row.folds),
            f"{row.mean_accuracy:.4f}",
            f"{row.mean_log_loss:.4f}",
            f"{row.mean_brier:.4f}",
        )

    console.print(table)
    console.print(f"Logistic tuning results written to: {output_path}")


@app.command("rolling-backtest")
def rolling_backtest_command(
    features_path: Annotated[
        Path,
        typer.Argument(help="Path to model-ready feature table CSV."),
    ] = DEFAULT_FEATURES_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for rolling backtest metrics CSV.",
        ),
    ] = DEFAULT_ROLLING_BACKTEST_METRICS_PATH,
    cutoff_dates_csv: Annotated[
        str,
        typer.Option(
            "--cutoff-dates",
            help="Comma-separated historical cutoff dates.",
        ),
    ] = ",".join(DEFAULT_ROLLING_CUTOFF_DATES),
    model_types_csv: Annotated[
        str,
        typer.Option(
            "--model-types",
            help="Comma-separated model types to compare.",
        ),
    ] = ",".join(DEFAULT_MODEL_TYPES),
    evaluation_window_days: Annotated[
        int,
        typer.Option(
            "--evaluation-window-days",
            help="Number of days after each cutoff to evaluate.",
        ),
    ] = 365,
    sample_weight_half_life_days: Annotated[
        float | None,
        typer.Option(
            "--sample-weight-half-life-days",
            help="Optional recency half-life in days for training weights.",
        ),
    ] = None,
) -> None:
    """Run rolling-origin model validation across historical cutoffs."""

    cutoff_dates = [
        cutoff.strip()
        for cutoff in cutoff_dates_csv.split(",")
        if cutoff.strip()
    ]
    model_types = [
        model_type.strip()
        for model_type in model_types_csv.split(",")
        if model_type.strip()
    ]

    results = save_rolling_backtest(
        features_path=features_path,
        output_path=output_path,
        cutoff_dates=cutoff_dates,
        model_types=model_types,
        evaluation_window_days=evaluation_window_days,
        sample_weight_half_life_days=sample_weight_half_life_days,
    )
    summary = summarize_rolling_backtest(results)

    table = Table(title="Rolling Backtest Model Comparison")
    table.add_column("Model")
    table.add_column("Folds", justify="right")
    table.add_column("Test Rows", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Log Loss", justify="right")
    table.add_column("Brier", justify="right")

    for row in summary.itertuples(index=False):
        table.add_row(
            str(row.model_type),
            str(row.folds),
            str(int(row.total_test_rows)),
            f"{row.mean_accuracy:.4f}",
            f"{row.mean_log_loss:.4f}",
            f"{row.mean_brier:.4f}",
        )

    console.print(table)
    console.print(f"Rolling backtest metrics written to: {output_path}")


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
    ] = 0.30,    cutoff_date: Annotated[
        str | None,
        typer.Option(
            "--cutoff-date",
            help="Optional cutoff date. Train before this date and test on/after it.",
        ),
    ] = None,
    sample_weight_half_life_days: Annotated[
        float | None,
        typer.Option(
            "--sample-weight-half-life-days",
            help=(
                "Optional recency half-life in days for time-decayed training "
                "weights."
            ),
        ),
    ] = None,
    model_type: Annotated[
        str,
        typer.Option(
            "--model-type",
            help="Model type: logistic, gradient_boosting, or random_forest.",
        ),
    ] = "logistic",
    logistic_c: Annotated[
        float,
        typer.Option(
            "--logistic-c",
            help="Inverse regularization strength for logistic model.",
        ),
    ] = DEFAULT_LOGISTIC_C,

) -> None:
    """Run a chronological logistic-regression backtest."""
    result = save_logistic_backtest(
        features_path=features_path,
        predictions_output_path=predictions_output,
        metrics_output_path=metrics_output,
        test_fraction=test_fraction,
        cutoff_date=cutoff_date,
        sample_weight_half_life_days=sample_weight_half_life_days,
        model_type=model_type,
        logistic_c=logistic_c,
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


@app.command("predict-poisson")
def predict_poisson(
    home_team: Annotated[
        str,
        typer.Argument(help="Home/team A name."),
    ],
    away_team: Annotated[
        str,
        typer.Argument(help="Away/team B name."),
    ],
    results_path: Annotated[
        Path,
        typer.Option(
            "--results-path",
            help="Path to processed historical results CSV.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for Poisson prediction CSV.",
        ),
    ] = DEFAULT_POISSON_PREDICTION_PATH,
    neutral: Annotated[
        bool,
        typer.Option(
            "--neutral/--not-neutral",
            help="Whether the match is played at a neutral site.",
        ),
    ] = True,
) -> None:
    """Generate an expected-goals Poisson forecast for one match."""
    results = load_historical_results(results_path)

    model = PoissonGoalsModel()
    model.fit(results)
    prediction = model.predict_match(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )

    save_poisson_prediction(
        results_path=results_path,
        home_team=home_team,
        away_team=away_team,
        output_path=output_path,
        neutral=neutral,
    )

    table = Table(title="Poisson Expected-Goals Forecast")
    table.add_column("Field")
    table.add_column("Value", justify="right")

    table.add_row("Match", f"{home_team} vs {away_team}")
    table.add_row("Expected home goals", f"{prediction.expected_home_goals:.3f}")
    table.add_row("Expected away goals", f"{prediction.expected_away_goals:.3f}")
    table.add_row("Home win probability", f"{prediction.prob_home_win:.3f}")
    table.add_row("Draw probability", f"{prediction.prob_draw:.3f}")
    table.add_row("Away win probability", f"{prediction.prob_away_win:.3f}")
    table.add_row("Most likely score", prediction.most_likely_score)
    table.add_row(
        "Most likely score probability",
        f"{prediction.most_likely_score_probability:.3f}",
    )

    console.print(table)
    console.print(f"[green]Poisson prediction written to:[/green] {output_path}")


@app.command("report-match")
def report_match(
    home_team: Annotated[
        str,
        typer.Argument(help="Home/team A name."),
    ],
    away_team: Annotated[
        str,
        typer.Argument(help="Away/team B name."),
    ],
    results_path: Annotated[
        Path,
        typer.Option(
            "--results-path",
            help="Path to processed historical results CSV.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    prediction_output: Annotated[
        Path,
        typer.Option(
            "--prediction-output",
            help="Path for combined match prediction CSV.",
        ),
    ] = DEFAULT_MATCH_PREDICTION_PATH,
    report_output: Annotated[
        Path,
        typer.Option(
            "--report-output",
            help="Path for Markdown match prediction report.",
        ),
    ] = DEFAULT_MATCH_REPORT_PATH,
    tournament: Annotated[
        str,
        typer.Option(
            "--tournament",
            help="Tournament or match context label.",
        ),
    ] = "FIFA World Cup",
    neutral: Annotated[
        bool,
        typer.Option(
            "--neutral/--not-neutral",
            help="Whether the match is played at a neutral site.",
        ),
    ] = True,
) -> None:
    """Generate an analyst-style match prediction report."""
    prediction_destination, report_destination = save_match_prediction_report(
        results_path=results_path,
        home_team=home_team,
        away_team=away_team,
        prediction_output_path=prediction_output,
        report_output_path=report_output,
        tournament=tournament,
        neutral=neutral,
    )

    console.print(f"[green]Match prediction written to:[/green] {prediction_destination}")
    console.print(f"[green]Match report written to:[/green] {report_destination}")







@app.command("list-forecast-artifacts")
def list_forecast_artifacts_command(
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for forecast artifact index CSV.",
        ),
    ] = DEFAULT_ARTIFACT_INDEX_PATH,
) -> None:
    """List generated forecast and validation artifacts."""

    artifact_index = save_artifact_index(output_path=output_path)

    table = Table(title="Forecast Artifact Index")
    table.add_column("Artifact", overflow="fold")
    table.add_column("Exists", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Modified At", overflow="fold")

    for row in artifact_index.to_dict("records"):
        table.add_row(
            str(row["path"]),
            "yes" if row["exists"] else "no",
            "" if pd.isna(row["size_bytes"]) else str(int(row["size_bytes"])),
            "" if pd.isna(row["modified_at"]) else str(row["modified_at"]),
        )

    console.print(table)
    console.print(f"Artifact index written to: {output_path}")


@app.command("summarize-upcoming-forecasts")
def summarize_upcoming_forecasts_command(
    forecasts_path: Annotated[
        Path,
        typer.Argument(help="Path to upcoming fixture forecast CSV."),
    ] = DEFAULT_WORLD_CUP_2026_UPCOMING_FORECASTS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for Markdown summary report.",
        ),
    ] = DEFAULT_WORLD_CUP_2026_UPCOMING_REPORT_PATH,
    max_rows: Annotated[
        int,
        typer.Option(
            "--max-rows",
            help="Maximum rows per report section.",
        ),
    ] = 10,
    upset_probability_threshold: Annotated[
        float,
        typer.Option(
            "--upset-threshold",
            help="Minimum lower-rated-team win probability for upset watch.",
        ),
    ] = 0.30,
) -> None:
    """Create a Markdown summary report from upcoming fixture forecasts."""

    if not forecasts_path.exists():
        console.print(f"[red]Forecast file not found:[/red] {forecasts_path}")
        console.print()
        console.print("Create it by running:")
        console.print("  python -m wc_forecast run-upcoming-world-cup-forecast")
        raise typer.Exit(code=1)

    save_upcoming_forecast_report(
        forecasts_path=forecasts_path,
        output_path=output_path,
        max_rows=max_rows,
        upset_probability_threshold=upset_probability_threshold,
    )

    console.print(f"Upcoming forecast report written to: {output_path}")


@app.command("run-upcoming-world-cup-forecast")
def run_upcoming_world_cup_forecast_command(
    raw_fixtures_path: Annotated[
        Path,
        typer.Option(
            "--raw-fixtures",
            help="Path to raw World Cup 2026 fixtures CSV.",
        ),
    ] = DEFAULT_WORLD_CUP_2026_RAW_FIXTURES_PATH,
    fixtures_path: Annotated[
        Path,
        typer.Option(
            "--fixtures",
            help="Path to processed World Cup 2026 fixtures CSV.",
        ),
    ] = DEFAULT_WORLD_CUP_2026_FIXTURES_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for upcoming fixture forecasts CSV.",
        ),
    ] = DEFAULT_WORLD_CUP_2026_UPCOMING_FORECASTS_PATH,
    from_date: Annotated[
        str | None,
        typer.Option(
            "--from-date",
            help="Forecast fixtures on or after this date. Defaults to today.",
        ),
    ] = None,
    through_date: Annotated[
        str | None,
        typer.Option(
            "--through-date",
            help="Optional final fixture date to include.",
        ),
    ] = None,
    train_cutoff_date: Annotated[
        str,
        typer.Option(
            "--train-cutoff-date",
            help="Train only on matches before this date.",
        ),
    ] = "2026-01-01",
    rating_cutoff_date: Annotated[
        str | None,
        typer.Option(
            "--rating-cutoff-date",
            help="Use ratings/form built only from results before this date.",
        ),
    ] = None,
    sample_weight_half_life_days: Annotated[
        float | None,
        typer.Option(
            "--sample-weight-half-life-days",
            help="Optional recency half-life in days for training weights.",
        ),
    ] = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    model_type: Annotated[
        str,
        typer.Option(
            "--model-type",
            help="Model type: logistic, gradient_boosting, or random_forest.",
        ),
    ] = "logistic",
    logistic_c: Annotated[
        float,
        typer.Option(
            "--logistic-c",
            help="Inverse regularization strength for logistic model.",
        ),
    ] = DEFAULT_LOGISTIC_C,
    skip_build_features: Annotated[
        bool,
        typer.Option(
            "--skip-build-features",
            help="Skip feature rebuilding before forecasting.",
        ),
    ] = False,
) -> None:
    """Run the full upcoming World Cup forecast workflow."""

    def run_step(label: str, args: list[str]) -> None:
        console.print()
        console.print(f"[bold]{label}[/bold]")
        subprocess.run(
            [sys.executable, "-m", "wc_forecast", *args],
            check=True,
        )

    if raw_fixtures_path.exists():
        run_step(
            "1. Ingest World Cup fixture schedule",
            [
                "ingest-world-cup-fixtures",
                str(raw_fixtures_path),
                "--output",
                str(fixtures_path),
            ],
        )
    elif fixtures_path.exists():
        console.print()
        console.print("1. Fixture schedule")
        console.print(f"Using existing processed fixture file: {fixtures_path}")
    else:
        console.print(f"[red]Fixture file not found:[/red] {fixtures_path}")
        console.print(f"[red]Raw fixture file not found:[/red] {raw_fixtures_path}")
        console.print()
        console.print("Create one of these files, then rerun this command:")
        console.print(f"  {raw_fixtures_path}")
        console.print(f"  {fixtures_path}")
        console.print()
        console.print("Or test with:")
        console.print(
            "  python -m wc_forecast forecast-upcoming-fixtures "
            "data/sample/world_cup_2026_fixtures_sample.csv"
        )
        raise typer.Exit(code=1)

    if skip_build_features:
        console.print()
        console.print("2. Feature build")
        console.print("Skipping feature rebuild.")
    else:
        run_step("2. Build model features", ["build-features"])

    forecast_args = [
        "forecast-upcoming-fixtures",
        str(fixtures_path),
        "--train-cutoff-date",
        train_cutoff_date,
        "--output",
        str(output_path),
        "--model-type",
        model_type,
        "--logistic-c",
        str(logistic_c),
    ]

    if from_date is not None:
        forecast_args.extend(["--from-date", from_date])

    if through_date is not None:
        forecast_args.extend(["--through-date", through_date])

    if rating_cutoff_date is not None:
        forecast_args.extend(["--rating-cutoff-date", rating_cutoff_date])

    if sample_weight_half_life_days is not None:
        forecast_args.extend(
            [
                "--sample-weight-half-life-days",
                str(sample_weight_half_life_days),
            ]
        )

    run_step("3. Forecast upcoming World Cup fixtures", forecast_args)

    run_step(
        "4. Summarize upcoming fixture forecasts",
        [
            "summarize-upcoming-forecasts",
            str(output_path),
            "--output",
            str(DEFAULT_WORLD_CUP_2026_UPCOMING_REPORT_PATH),
        ],
    )

    run_step(
        "5. Index generated forecast artifacts",
        [
            "list-forecast-artifacts",
            "--output",
            str(DEFAULT_ARTIFACT_INDEX_PATH),
        ],
    )

    console.print()
    console.print("[green]Upcoming World Cup forecast workflow complete.[/green]")
    console.print(f"Forecast output: {output_path}")
    console.print(f"Report output: {DEFAULT_WORLD_CUP_2026_UPCOMING_REPORT_PATH}")
    console.print(f"Artifact index: {DEFAULT_ARTIFACT_INDEX_PATH}")


@app.command("ingest-world-cup-fixtures")
def ingest_world_cup_fixtures_command(
    source_path: Annotated[
        Path,
        typer.Argument(help="Path to raw World Cup 2026 fixtures CSV."),
    ] = DEFAULT_WORLD_CUP_2026_RAW_FIXTURES_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for normalized processed World Cup fixtures CSV.",
        ),
    ] = DEFAULT_WORLD_CUP_2026_FIXTURES_PATH,
) -> None:
    """Normalize a raw World Cup fixture schedule for forecasting."""

    fixtures = save_world_cup_fixtures(
        source_path=source_path,
        output_path=output_path,
    )

    console.print(f"Processed {len(fixtures)} World Cup fixtures.")
    console.print(f"Fixture schedule written to: {output_path}")


@app.command("forecast-upcoming-fixtures")
def forecast_upcoming_fixtures_command(
    fixtures_path: Annotated[
        Path,
        typer.Argument(help="Path to full World Cup 2026 fixtures CSV."),
    ] = DEFAULT_WORLD_CUP_2026_FIXTURES_PATH,
    features_path: Annotated[
        Path,
        typer.Option(
            "--features",
            help="Path to model-ready feature table CSV.",
        ),
    ] = DEFAULT_FEATURES_PATH,
    results_path: Annotated[
        Path,
        typer.Option(
            "--results",
            help="Path to processed historical results CSV.",
        ),
    ] = DEFAULT_UPCOMING_FORECAST_RESULTS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for upcoming fixture forecasts CSV.",
        ),
    ] = DEFAULT_WORLD_CUP_2026_UPCOMING_FORECASTS_PATH,
    train_cutoff_date: Annotated[
        str,
        typer.Option(
            "--train-cutoff-date",
            help="Train only on matches before this date.",
        ),
    ] = "2026-01-01",
    from_date: Annotated[
        str | None,
        typer.Option(
            "--from-date",
            help="Forecast fixtures on or after this date. Defaults to today.",
        ),
    ] = None,
    through_date: Annotated[
        str | None,
        typer.Option(
            "--through-date",
            help="Optional final fixture date to include.",
        ),
    ] = None,
    rating_cutoff_date: Annotated[
        str | None,
        typer.Option(
            "--rating-cutoff-date",
            help="Use ratings/form built only from results before this date.",
        ),
    ] = None,
    sample_weight_half_life_days: Annotated[
        float | None,
        typer.Option(
            "--sample-weight-half-life-days",
            help="Optional recency half-life in days for training weights.",
        ),
    ] = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    model_type: Annotated[
        str,
        typer.Option(
            "--model-type",
            help="Model type: logistic, gradient_boosting, or random_forest.",
        ),
    ] = "logistic",
    logistic_c: Annotated[
        float,
        typer.Option(
            "--logistic-c",
            help="Inverse regularization strength for logistic model.",
        ),
    ] = DEFAULT_LOGISTIC_C,
    include_tbd: Annotated[
        bool,
        typer.Option(
            "--include-tbd",
            help="Include TBD knockout placeholders if present.",
        ),
    ] = False,
) -> None:
    """Automatically forecast all upcoming known-team World Cup fixtures."""

    if not fixtures_path.exists():
        console.print(f"[red]Fixture file not found:[/red] {fixtures_path}")
        console.print()
        console.print("Create it by running:")
        console.print("  python -m wc_forecast ingest-world-cup-fixtures \\")
        console.print("    data/raw/world_cup_2026_fixtures.csv \\")
        console.print("    --output data/processed/world_cup_2026_fixtures.csv")
        console.print()
        console.print("Or test with the included sample:")
        console.print("  python -m wc_forecast forecast-upcoming-fixtures \\")
        console.print("    data/sample/world_cup_2026_fixtures_sample.csv")
        raise typer.Exit(code=1)

    forecasts = save_upcoming_fixture_forecasts_from_results(
        fixtures_path=fixtures_path,
        features_path=features_path,
        results_path=results_path,
        output_path=output_path,
        train_cutoff_date=train_cutoff_date,
        from_date=from_date,
        through_date=through_date,
        rating_cutoff_date=rating_cutoff_date,
        sample_weight_half_life_days=sample_weight_half_life_days,
        model_type=model_type,
        logistic_c=logistic_c,
        include_tbd=include_tbd,
    )

    console.print(f"Forecasted {len(forecasts)} upcoming fixtures.")
    console.print(f"Upcoming fixture forecasts written to: {output_path}")


@app.command("forecast-fixtures")
def forecast_fixtures_command(
    fixtures_path: Annotated[
        Path,
        typer.Argument(help="Path to fixture slate CSV."),
    ] = DEFAULT_WORLD_CUP_FIXTURES_PATH,
    features_path: Annotated[
        Path,
        typer.Option(
            "--features-path",
            help="Path to model-ready historical feature table CSV.",
        ),
    ] = DEFAULT_FEATURES_PATH,
    results_path: Annotated[
        Path,
        typer.Option(
            "--results-path",
            help="Path to processed historical results CSV used to build ratings.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for fixture forecast CSV.",
        ),
    ] = DEFAULT_WORLD_CUP_FORECASTS_PATH,
    train_cutoff_date: Annotated[
        str,
        typer.Option(
            "--train-cutoff-date",
            help="Train only on historical feature rows before this date.",
        ),
    ] = "2026-01-01",
    rating_cutoff_date: Annotated[
        str | None,
        typer.Option(
            "--rating-cutoff-date",
            help=(
                "Build Elo ratings only from result rows before this date. "
                "Defaults to train cutoff date."
            ),
        ),
    ] = None,
    sample_weight_half_life_days: Annotated[
        float | None,
        typer.Option(
            "--sample-weight-half-life-days",
            help=(
                "Optional recency half-life in days for time-decayed training "
                "weights."
            ),
        ),
    ] = None,
    model_type: Annotated[
        str,
        typer.Option(
            "--model-type",
            help="Model type: logistic, gradient_boosting, or random_forest.",
        ),
    ] = "logistic",
    logistic_c: Annotated[
        float,
        typer.Option(
            "--logistic-c",
            help="Inverse regularization strength for logistic model.",
        ),
    ] = DEFAULT_LOGISTIC_C,
) -> None:
    """Forecast a slate of FIFA World Cup fixtures."""
    forecasts = save_fixture_forecasts_from_results(
        fixtures_path=fixtures_path,
        features_path=features_path,
        results_path=results_path,
        output_path=output_path,
        train_cutoff_date=train_cutoff_date,
        rating_cutoff_date=rating_cutoff_date,
        sample_weight_half_life_days=sample_weight_half_life_days,
        model_type=model_type,
        logistic_c=logistic_c,
    )

    table = Table(title="Fixture Forecasts")
    table.add_column("Match")
    table.add_column("Predicted Winner")
    table.add_column("Home", justify="right")
    table.add_column("Draw", justify="right")
    table.add_column("Away", justify="right")
    table.add_column("Confidence", justify="right")

    for row in forecasts.itertuples(index=False):
        table.add_row(
            f"{row.home_team} vs {row.away_team}",
            row.predicted_winner,
            f"{row.prob_home_win:.1%}",
            f"{row.prob_draw:.1%}",
            f"{row.prob_away_win:.1%}",
            f"{row.model_confidence:.1%}",
        )

    console.print(table)
    console.print(f"[green]Fixture forecasts written to:[/green] {output_path}")


@app.command("simulate-group-stage")
def simulate_group_stage_command(
    results_path: Annotated[
        Path,
        typer.Option(
            "--results-path",
            help="Path to processed historical results CSV.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    fixtures_path: Annotated[
        Path,
        typer.Option(
            "--fixtures-path",
            help="Path to group-stage fixture CSV.",
        ),
    ] = DEFAULT_GROUP_FIXTURES_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for group-stage simulation summary CSV.",
        ),
    ] = DEFAULT_GROUP_SIMULATION_PATH,
    n_simulations: Annotated[
        int,
        typer.Option(
            "--n-simulations",
            help="Number of Monte Carlo simulations to run.",
        ),
    ] = 1_000,
    qualifiers_per_group: Annotated[
        int,
        typer.Option(
            "--qualifiers-per-group",
            help="Number of teams advancing from each group.",
        ),
    ] = 2,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducible simulation.",
        ),
    ] = 42,
) -> None:
    """Run Monte Carlo group-stage simulation."""
    destination = save_group_stage_simulation(
        results_path=results_path,
        fixtures_path=fixtures_path,
        output_path=output_path,
        n_simulations=n_simulations,
        qualifiers_per_group=qualifiers_per_group,
        seed=seed,
    )

    console.print(f"[green]Group-stage simulation written to:[/green] {destination}")


@app.command("report-group-stage")
def report_group_stage(
    summary_path: Annotated[
        Path,
        typer.Option(
            "--summary-path",
            help="Path to group-stage simulation summary CSV.",
        ),
    ] = DEFAULT_GROUP_SIMULATION_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for Markdown group-stage simulation report.",
        ),
    ] = DEFAULT_GROUP_SIMULATION_REPORT_PATH,
) -> None:
    """Generate a Markdown report from group-stage simulation outputs."""
    destination = save_group_stage_report(
        summary_path=summary_path,
        output_path=output_path,
    )

    console.print(f"[green]Group-stage report written to:[/green] {destination}")


@app.command("evaluate-market")
def evaluate_market(
    home_team: Annotated[
        str,
        typer.Argument(help="Home/team A name."),
    ],
    away_team: Annotated[
        str,
        typer.Argument(help="Away/team B name."),
    ],
    home_odds: Annotated[
        float,
        typer.Option(
            "--home-odds",
            help="Decimal odds for home/team A win.",
        ),
    ],
    draw_odds: Annotated[
        float,
        typer.Option(
            "--draw-odds",
            help="Decimal odds for draw.",
        ),
    ],
    away_odds: Annotated[
        float,
        typer.Option(
            "--away-odds",
            help="Decimal odds for away/team B win.",
        ),
    ],
    results_path: Annotated[
        Path,
        typer.Option(
            "--results-path",
            help="Path to processed historical results CSV.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for market edge evaluation CSV.",
        ),
    ] = DEFAULT_MARKET_EDGE_PATH,
    minimum_edge: Annotated[
        float,
        typer.Option(
            "--minimum-edge",
            help="Minimum model-vs-market probability edge.",
        ),
    ] = 0.03,
    minimum_expected_value: Annotated[
        float,
        typer.Option(
            "--minimum-expected-value",
            help="Minimum expected value threshold.",
        ),
    ] = 0.02,
) -> None:
    """Compare ensemble probabilities against market odds."""
    results = load_historical_results(results_path)
    prediction = generate_match_prediction(
        results=results,
        home_team=home_team,
        away_team=away_team,
    )

    market = calculate_market_probabilities(
        home_win_odds=home_odds,
        draw_odds=draw_odds,
        away_win_odds=away_odds,
    )
    edge = calculate_market_edge(
        model_prob_home_win=float(prediction["ensemble_prob_home_win"]),
        model_prob_draw=float(prediction["ensemble_prob_draw"]),
        model_prob_away_win=float(prediction["ensemble_prob_away_win"]),
        home_win_odds=home_odds,
        draw_odds=draw_odds,
        away_win_odds=away_odds,
        minimum_edge=minimum_edge,
        minimum_expected_value=minimum_expected_value,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "market_overround": market.overround,
                "market_fair_home_win": market.fair_home_win,
                "market_fair_draw": market.fair_draw,
                "market_fair_away_win": market.fair_away_win,
                "model_prob_home_win": prediction["ensemble_prob_home_win"],
                "model_prob_draw": prediction["ensemble_prob_draw"],
                "model_prob_away_win": prediction["ensemble_prob_away_win"],
                "edge_home_win": edge.edge_home_win,
                "edge_draw": edge.edge_draw,
                "edge_away_win": edge.edge_away_win,
                "expected_value_home_win": edge.expected_value_home_win,
                "expected_value_draw": edge.expected_value_draw,
                "expected_value_away_win": edge.expected_value_away_win,
                "best_outcome": edge.best_outcome,
                "best_edge": edge.best_edge,
                "best_expected_value": edge.best_expected_value,
                "decision": edge.decision,
            }
        ]
    ).to_csv(output_path, index=False)

    table = Table(title="Market Edge Evaluation")
    table.add_column("Field")
    table.add_column("Value", justify="right")

    model_home_probability = float(prediction["ensemble_prob_home_win"])
    model_draw_probability = float(prediction["ensemble_prob_draw"])
    model_away_probability = float(prediction["ensemble_prob_away_win"])

    table.add_row("Match", f"{home_team} vs {away_team}")
    table.add_row("Market overround", f"{market.overround:.3f}")
    table.add_row("Model home win probability", f"{model_home_probability:.3f}")
    table.add_row("Market fair home win probability", f"{market.fair_home_win:.3f}")
    table.add_row("Model draw probability", f"{model_draw_probability:.3f}")
    table.add_row("Market fair draw probability", f"{market.fair_draw:.3f}")
    table.add_row("Model away win probability", f"{model_away_probability:.3f}")
    table.add_row("Market fair away win probability", f"{market.fair_away_win:.3f}")
    table.add_row("Best outcome", edge.best_outcome)
    table.add_row("Best edge", f"{edge.best_edge:.3f}")
    table.add_row("Best expected value", f"{edge.best_expected_value:.3f}")
    table.add_row("Decision", edge.decision)

    console.print(table)
    console.print(f"[green]Market edge evaluation written to:[/green] {output_path}")


@app.command("log-prediction")
def log_prediction(
    home_team: Annotated[
        str,
        typer.Argument(help="Home/team A name."),
    ],
    away_team: Annotated[
        str,
        typer.Argument(help="Away/team B name."),
    ],
    home_odds: Annotated[
        float,
        typer.Option(
            "--home-odds",
            help="Decimal odds for home/team A win.",
        ),
    ],
    draw_odds: Annotated[
        float,
        typer.Option(
            "--draw-odds",
            help="Decimal odds for draw.",
        ),
    ],
    away_odds: Annotated[
        float,
        typer.Option(
            "--away-odds",
            help="Decimal odds for away/team B win.",
        ),
    ],
    results_path: Annotated[
        Path,
        typer.Option(
            "--results-path",
            help="Path to processed historical results CSV.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    ledger_path: Annotated[
        Path,
        typer.Option(
            "--ledger-path",
            help="Path to prediction ledger CSV.",
        ),
    ] = DEFAULT_PREDICTION_LEDGER_PATH,
    model_version: Annotated[
        str,
        typer.Option(
            "--model-version",
            help="Model version label for audit tracking.",
        ),
    ] = "demo-v1",
    feature_version: Annotated[
        str,
        typer.Option(
            "--feature-version",
            help="Feature version label for audit tracking.",
        ),
    ] = "demo-features-v1",
    prediction_timestamp: Annotated[
        str | None,
        typer.Option(
            "--prediction-timestamp",
            help="Optional prediction timestamp. Defaults to current UTC time.",
        ),
    ] = None,
    kickoff_timestamp: Annotated[
        str,
        typer.Option(
            "--kickoff-timestamp",
            help="Optional kickoff timestamp.",
        ),
    ] = "",
    notes: Annotated[
        str,
        typer.Option(
            "--notes",
            help="Optional audit notes.",
        ),
    ] = "",
) -> None:
    """Append a market-aware forecast to the prediction ledger."""
    destination = save_market_prediction_to_ledger(
        results_path=results_path,
        ledger_path=ledger_path,
        home_team=home_team,
        away_team=away_team,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        model_version=model_version,
        feature_version=feature_version,
        prediction_timestamp=prediction_timestamp,
        kickoff_timestamp=kickoff_timestamp,
        notes=notes,
    )

    console.print(f"[green]Prediction logged to ledger:[/green] {destination}")


@app.command("settle-prediction")
def settle_prediction(
    prediction_id: Annotated[
        str,
        typer.Argument(help="Prediction ID from the ledger."),
    ],
    final_home_score: Annotated[
        int,
        typer.Option(
            "--final-home-score",
            help="Final home/team A score.",
        ),
    ],
    final_away_score: Annotated[
        int,
        typer.Option(
            "--final-away-score",
            help="Final away/team B score.",
        ),
    ],
    ledger_path: Annotated[
        Path,
        typer.Option(
            "--ledger-path",
            help="Path to prediction ledger CSV.",
        ),
    ] = DEFAULT_PREDICTION_LEDGER_PATH,
    closing_home_odds: Annotated[
        float | None,
        typer.Option(
            "--closing-home-odds",
            help="Optional closing decimal odds for home/team A win.",
        ),
    ] = None,
    closing_draw_odds: Annotated[
        float | None,
        typer.Option(
            "--closing-draw-odds",
            help="Optional closing decimal odds for draw.",
        ),
    ] = None,
    closing_away_odds: Annotated[
        float | None,
        typer.Option(
            "--closing-away-odds",
            help="Optional closing decimal odds for away/team B win.",
        ),
    ] = None,
    stake: Annotated[
        float,
        typer.Option(
            "--stake",
            help="Flat stake used to calculate realized return.",
        ),
    ] = 1.0,
) -> None:
    """Settle a logged prediction with final result and realized return."""
    destination = settle_prediction_ledger_row(
        ledger_path=ledger_path,
        prediction_id=prediction_id,
        final_home_score=final_home_score,
        final_away_score=final_away_score,
        closing_home_odds=closing_home_odds,
        closing_draw_odds=closing_draw_odds,
        closing_away_odds=closing_away_odds,
        stake=stake,
    )

    console.print(f"[green]Prediction ledger settled:[/green] {destination}")


@app.command("report-ledger")
def report_ledger(
    ledger_path: Annotated[
        Path,
        typer.Option(
            "--ledger-path",
            help="Path to prediction ledger CSV.",
        ),
    ] = DEFAULT_PREDICTION_LEDGER_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for Markdown prediction ledger report.",
        ),
    ] = DEFAULT_PREDICTION_LEDGER_REPORT_PATH,
) -> None:
    """Generate a Markdown performance report from the prediction ledger."""
    destination = save_prediction_ledger_report(
        ledger_path=ledger_path,
        output_path=output_path,
    )

    console.print(f"[green]Prediction ledger report written to:[/green] {destination}")


@app.command("batch-evaluate-market")
def batch_evaluate_market(
    odds_path: Annotated[
        Path,
        typer.Argument(help="Path to market odds slate CSV."),
    ] = DEFAULT_MARKET_ODDS_PATH,
    results_path: Annotated[
        Path,
        typer.Option(
            "--results-path",
            help="Path to processed historical results CSV.",
        ),
    ] = DEFAULT_PROCESSED_RESULTS_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for batch market edge CSV.",
        ),
    ] = DEFAULT_BATCH_MARKET_EDGE_PATH,
    minimum_edge: Annotated[
        float,
        typer.Option(
            "--minimum-edge",
            help="Minimum model-vs-market probability edge.",
        ),
    ] = 0.03,
    minimum_expected_value: Annotated[
        float,
        typer.Option(
            "--minimum-expected-value",
            help="Minimum expected value for candidate edge.",
        ),
    ] = 0.02,
) -> None:
    """Evaluate a slate of fixtures and market odds."""
    results = load_historical_results(results_path)
    odds = pd.read_csv(odds_path)

    destination = save_market_odds_slate_evaluation(
        results=results,
        odds=odds,
        output_path=output_path,
        minimum_edge=minimum_edge,
        minimum_expected_value=minimum_expected_value,
    )

    evaluation = pd.read_csv(destination)

    table = Table(title="Batch Market Edge Evaluation")
    table.add_column("Match")
    table.add_column("Decision", justify="right")
    table.add_column("Best Outcome", justify="right")
    table.add_column("Best Edge", justify="right")
    table.add_column("Best EV", justify="right")

    for row in evaluation.head(10).itertuples(index=False):
        table.add_row(
            f"{row.home_team} vs {row.away_team}",
            str(row.decision),
            str(row.best_outcome),
            f"{float(row.best_edge):.3f}",
            f"{float(row.best_expected_value):.3f}",
        )

    console.print(table)
    console.print(f"[green]Batch market evaluation written to:[/green] {destination}")


@app.command("log-batch-predictions")
def log_batch_predictions(
    batch_edges_path: Annotated[
        Path,
        typer.Argument(help="Path to batch market edge CSV."),
    ] = DEFAULT_BATCH_MARKET_EDGE_PATH,
    ledger_path: Annotated[
        Path,
        typer.Option(
            "--ledger-path",
            help="Path to prediction ledger CSV.",
        ),
    ] = DEFAULT_PREDICTION_LEDGER_PATH,
    model_version: Annotated[
        str,
        typer.Option(
            "--model-version",
            help="Model version label for audit tracking.",
        ),
    ] = "demo-v1",
    feature_version: Annotated[
        str,
        typer.Option(
            "--feature-version",
            help="Feature version label for audit tracking.",
        ),
    ] = "demo-features-v1",
    prediction_timestamp: Annotated[
        str | None,
        typer.Option(
            "--prediction-timestamp",
            help="Optional prediction timestamp. Defaults to current UTC time.",
        ),
    ] = None,
    notes: Annotated[
        str,
        typer.Option(
            "--notes",
            help="Optional audit notes.",
        ),
    ] = "",
) -> None:
    """Append candidate edges from a batch market evaluation to the ledger."""
    destination = append_candidate_edges_to_prediction_ledger(
        batch_edges_path=batch_edges_path,
        ledger_path=ledger_path,
        model_version=model_version,
        feature_version=feature_version,
        prediction_timestamp=prediction_timestamp,
        notes=notes,
    )

    console.print(f"[green]Batch predictions logged to ledger:[/green] {destination}")


@app.command("settle-batch-predictions")
def settle_batch_predictions(
    settlement_results_path: Annotated[
        Path,
        typer.Argument(help="Path to settlement results CSV."),
    ] = DEFAULT_SETTLEMENT_RESULTS_PATH,
    ledger_path: Annotated[
        Path,
        typer.Option(
            "--ledger-path",
            help="Path to prediction ledger CSV.",
        ),
    ] = DEFAULT_PREDICTION_LEDGER_PATH,
    stake: Annotated[
        float,
        typer.Option(
            "--stake",
            help="Flat stake used to calculate realized return.",
        ),
    ] = 1.0,
) -> None:
    """Settle matching ledger predictions from a results CSV."""
    destination = settle_prediction_ledger_from_results(
        ledger_path=ledger_path,
        settlement_results_path=settlement_results_path,
        stake=stake,
    )

    console.print(f"[green]Batch predictions settled in ledger:[/green] {destination}")


@app.command("apply-strategy-policy")
def apply_strategy_policy_command(
    batch_edges_path: Annotated[
        Path,
        typer.Argument(help="Path to batch market edge CSV."),
    ] = DEFAULT_BATCH_MARKET_EDGE_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for strategy policy output CSV.",
        ),
    ] = DEFAULT_STRATEGY_POLICY_PATH,
    minimum_edge: Annotated[
        float,
        typer.Option(
            "--minimum-edge",
            help="Minimum model-vs-market probability edge.",
        ),
    ] = 0.05,
    minimum_expected_value: Annotated[
        float,
        typer.Option(
            "--minimum-expected-value",
            help="Minimum expected value.",
        ),
    ] = 0.05,
    maximum_entropy: Annotated[
        float,
        typer.Option(
            "--maximum-entropy",
            help="Maximum ensemble entropy.",
        ),
    ] = 1.00,
    maximum_model_disagreement: Annotated[
        float,
        typer.Option(
            "--maximum-model-disagreement",
            help="Maximum disagreement across model components.",
        ),
    ] = 0.50,
    maximum_market_overround: Annotated[
        float,
        typer.Option(
            "--maximum-market-overround",
            help="Maximum allowed market overround.",
        ),
    ] = 1.08,
) -> None:
    """Apply strategy policy gates to ranked market edges."""
    policy = StrategyPolicy(
        minimum_edge=minimum_edge,
        minimum_expected_value=minimum_expected_value,
        maximum_entropy=maximum_entropy,
        maximum_model_disagreement=maximum_model_disagreement,
        maximum_market_overround=maximum_market_overround,
    )

    destination = save_strategy_policy_output(
        batch_edges_path=batch_edges_path,
        output_path=output_path,
        policy=policy,
    )

    policy_output = pd.read_csv(destination)

    table = Table(title="Strategy Policy Decisions")
    table.add_column("Match")
    table.add_column("Action", justify="right")
    table.add_column("Best Outcome", justify="right")
    table.add_column("Best EV", justify="right")
    table.add_column("Reason")

    for row in policy_output.head(10).itertuples(index=False):
        table.add_row(
            f"{row.home_team} vs {row.away_team}",
            str(row.strategy_action),
            str(row.best_outcome),
            f"{float(row.best_expected_value):.3f}",
            str(row.strategy_reason),
        )

    console.print(table)
    console.print(f"[green]Strategy policy output written to:[/green] {destination}")


@app.command("size-stakes")
def size_stakes_command(
    strategy_policy_path: Annotated[
        Path,
        typer.Argument(help="Path to strategy policy CSV."),
    ] = DEFAULT_STRATEGY_POLICY_PATH,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for stake sizing output CSV.",
        ),
    ] = DEFAULT_STAKE_SIZING_PATH,
    bankroll: Annotated[
        float,
        typer.Option(
            "--bankroll",
            help="Bankroll used to convert fractions into stake amounts.",
        ),
    ] = 1000.0,
    fractional_kelly: Annotated[
        float,
        typer.Option(
            "--fractional-kelly",
            help="Fraction of full Kelly to use.",
        ),
    ] = 0.25,
    max_single_bet_fraction: Annotated[
        float,
        typer.Option(
            "--max-single-bet-fraction",
            help="Maximum stake fraction for one prediction.",
        ),
    ] = 0.02,
    max_portfolio_exposure_fraction: Annotated[
        float,
        typer.Option(
            "--max-portfolio-exposure-fraction",
            help="Maximum total stake fraction across the slate.",
        ),
    ] = 0.05,
) -> None:
    """Apply fractional Kelly stake sizing to actionable edges."""
    policy = StakeSizingPolicy(
        bankroll=bankroll,
        fractional_kelly=fractional_kelly,
        max_single_bet_fraction=max_single_bet_fraction,
        max_portfolio_exposure_fraction=max_portfolio_exposure_fraction,
    )

    destination = save_stake_sizing_output(
        strategy_policy_path=strategy_policy_path,
        output_path=output_path,
        policy=policy,
    )

    stake_output = pd.read_csv(destination)

    table = Table(title="Stake Sizing Decisions")
    table.add_column("Match")
    table.add_column("Action", justify="right")
    table.add_column("Outcome", justify="right")
    table.add_column("Stake %", justify="right")
    table.add_column("Stake Amount", justify="right")
    table.add_column("Reason")

    for row in stake_output.head(10).itertuples(index=False):
        table.add_row(
            f"{row.home_team} vs {row.away_team}",
            str(row.strategy_action),
            str(row.best_outcome),
            f"{float(row.suggested_stake_fraction) * 100:.2f}%",
            f"{float(row.suggested_stake_amount):.2f}",
            str(row.stake_sizing_reason),
        )

    console.print(table)
    console.print(f"[green]Stake sizing output written to:[/green] {destination}")


@app.command("ingest-real-results")
def ingest_real_results(
    source_path: Annotated[
        Path,
        typer.Argument(help="Path to real international results CSV."),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for normalized real results CSV.",
        ),
    ] = DEFAULT_REAL_RESULTS_PATH,
) -> None:
    """Normalize a real international-results CSV into the engine schema."""
    destination = save_normalized_international_results(
        source_path=source_path,
        output_path=output_path,
    )

    console.print(f"[green]Real international results written to:[/green] {destination}")


if __name__ == "__main__":
    app()
