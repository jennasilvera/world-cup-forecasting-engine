from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_PREDICTION_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "tournament",
    "home_score",
    "away_score",
    "actual_outcome",
    "predicted_outcome",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
]

REQUIRED_METRIC_COLUMNS = [
    "metric",
    "value",
]

PROBABILITY_COLUMNS = [
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
]


def _format_probability(value: float) -> str:
    """Format a probability as a percentage string."""

    return f"{value * 100:.1f}%"


def _format_metric_value(metric: str, value: float) -> str:
    """Format metric values for a Markdown report."""

    if metric.endswith("rows"):
        return str(int(round(value)))

    return f"{value:.4f}"


def validate_backtest_inputs(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Validate model prediction and metric tables before report generation."""

    missing_prediction_columns = sorted(
        set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    )
    if missing_prediction_columns:
        raise ValueError(
            "Prediction table missing required columns: "
            f"{missing_prediction_columns}"
        )

    missing_metric_columns = sorted(set(REQUIRED_METRIC_COLUMNS) - set(metrics.columns))
    if missing_metric_columns:
        raise ValueError(f"Metric table missing required columns: {missing_metric_columns}")

    if predictions.empty:
        raise ValueError("Prediction table is empty.")

    if metrics.empty:
        raise ValueError("Metric table is empty.")

    if predictions[PROBABILITY_COLUMNS].isna().any().any():
        raise ValueError("Prediction table contains missing probability values.")

    probability_sums = predictions[PROBABILITY_COLUMNS].sum(axis=1)
    invalid_probability_rows = ~probability_sums.round(6).eq(1.0)

    if invalid_probability_rows.any():
        raise ValueError("Prediction probabilities must sum to 1.0 for every row.")


def render_backtest_report(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    model_name: str = "Logistic Regression Baseline",
    max_matches: int = 10,
) -> str:
    """Render a recruiter-facing Markdown report from backtest outputs."""

    validate_backtest_inputs(predictions=predictions, metrics=metrics)

    metrics_lookup = {
        str(row.metric): float(row.value)
        for row in metrics.itertuples(index=False)
    }

    metric_rows = [
        f"| {row.metric} | {_format_metric_value(str(row.metric), float(row.value))} |"
        for row in metrics.itertuples(index=False)
    ]

    preview = predictions.tail(max_matches).copy()
    preview["date"] = pd.to_datetime(preview["date"], errors="raise").dt.date

    match_rows: list[str] = []

    for row in preview.itertuples(index=False):
        match = f"{row.home_team} vs {row.away_team}"
        score = f"{int(row.home_score)}-{int(row.away_score)}"
        probability_summary = (
            f"H {_format_probability(float(row.prob_home_win))} / "
            f"D {_format_probability(float(row.prob_draw))} / "
            f"A {_format_probability(float(row.prob_away_win))}"
        )

        match_rows.append(
            "| "
            f"{row.date} | {match} | {score} | {row.actual_outcome} | "
            f"{row.predicted_outcome} | {probability_summary} |"
        )

    accuracy = metrics_lookup.get("accuracy")
    log_loss = metrics_lookup.get("log_loss")
    brier_score = metrics_lookup.get("multiclass_brier_score")

    summary_parts = []

    if accuracy is not None:
        summary_parts.append(f"accuracy of {accuracy:.4f}")

    if log_loss is not None:
        summary_parts.append(f"log loss of {log_loss:.4f}")

    if brier_score is not None:
        summary_parts.append(f"multiclass Brier score of {brier_score:.4f}")

    summary_sentence = ", ".join(summary_parts)

    return f"""# Backtest Report: {model_name}

## Purpose

This report summarizes a chronological backtest for the current World Cup Match
Forecasting Engine model.

The model is evaluated as a probabilistic forecasting system, not as a perfect
match-result predictor. The goal is to estimate pre-match win/draw/loss
probabilities and evaluate whether those probabilities are reasonable,
transparent, and testable.

## Backtest Summary

The latest backtest produced {summary_sentence}.

The latest run may use either the small repository sample dataset or a larger
real international-results dataset, depending on the local processed data file.
Metric values should be interpreted in the context of the data source, feature
set, train/test split, and known model limitations. The report demonstrates that
the pipeline can train, predict, score, and generate auditable outputs
end-to-end.

## Metrics

| Metric | Value |
|---|---:|
{chr(10).join(metric_rows)}

## Recent Match-Level Predictions

| Date | Match | Score | Actual Outcome | Predicted Outcome | Probabilities |
|---|---|---:|---|---|---|
{chr(10).join(match_rows)}

## Model Inputs

The current baseline uses pre-match features generated from custom Elo ratings
and match context fields:

- Home-team pre-match Elo rating
- Away-team pre-match Elo rating
- Elo rating difference
- Absolute Elo rating difference
- Elo expected home/away score
- Neutral-site flag
- World Cup match flag
- Tournament importance weight

## Leakage Controls

The feature table is built chronologically. Elo ratings are captured before each
match update, and final scores are stored separately as result/target columns.
This prevents final scores from leaking into pre-match features.

## Current Limitations

- The baseline can run on either sample data or a larger real international
  results dataset.
- The current backtest uses a simple chronological split and should be upgraded
  with explicit train/evaluation cutoff dates.
- Current features are mostly team-strength and match-context features.
- Player availability, injuries, travel, rest, and market odds are planned
  future upgrades.
- Logistic regression is used as an interpretable baseline, not the final model.

## Next Improvements

- Add larger public historical results data.
- Add rolling form features.
- Add Poisson expected-goals model.
- Add calibrated ensemble forecasts.
- Add Monte Carlo tournament simulation.
- Add calibration plots and model cards.
"""


def save_backtest_report(
    predictions_path: str | Path,
    metrics_path: str | Path,
    output_path: str | Path,
    model_name: str = "Logistic Regression Baseline",
) -> Path:
    """Load backtest outputs and save a Markdown report."""

    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)

    report = render_backtest_report(
        predictions=predictions,
        metrics=metrics,
        model_name=model_name,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report)

    return destination
