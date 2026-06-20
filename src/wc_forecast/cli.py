from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from wc_forecast.data.ingest_results import load_historical_results, save_processed_results
from wc_forecast.features.build_features import save_match_features
from wc_forecast.ledger.prediction_ledger import (
    save_market_prediction_to_ledger,
    settle_prediction_ledger_row,
)
from wc_forecast.models.classifier import save_logistic_backtest
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
from wc_forecast.simulation.group_stage import save_group_stage_simulation

DEFAULT_PROCESSED_RESULTS_PATH = Path("data/processed/results.csv")
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
DEFAULT_GROUP_SIMULATION_PATH = Path("outputs/group_stage_simulation.csv")
DEFAULT_GROUP_SIMULATION_REPORT_PATH = Path("reports/group_stage_simulation_report.md")
DEFAULT_MARKET_EDGE_PATH = Path("outputs/market_edge.csv")
DEFAULT_PREDICTION_LEDGER_PATH = Path("outputs/prediction_ledger.csv")
DEFAULT_PREDICTION_LEDGER_REPORT_PATH = Path("outputs/prediction_ledger_report.md")

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


if __name__ == "__main__":
    app()
