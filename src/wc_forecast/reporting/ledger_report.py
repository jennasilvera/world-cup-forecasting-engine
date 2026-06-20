from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_LEDGER_REPORT_COLUMNS = [
    "prediction_id",
    "home_team",
    "away_team",
    "decision",
    "best_outcome",
    "best_edge",
    "best_expected_value",
    "home_odds",
    "draw_odds",
    "away_odds",
    "closing_home_odds",
    "closing_draw_odds",
    "closing_away_odds",
    "final_outcome",
    "realized_return",
]


def _has_value(value: object) -> bool:
    """Return whether a CSV value is meaningfully populated."""

    return str(value).strip() != "" and str(value).strip().lower() != "nan"


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert CSV values to float with a safe default."""

    if not _has_value(value):
        return default

    return float(value)


def _format_probability(value: float) -> str:
    """Format a decimal probability as a percentage."""

    return f"{value * 100:.1f}%"


def _format_money_return(value: float) -> str:
    """Format flat-stake return."""

    return f"{value:.3f}"


def _selected_opening_odds(row: pd.Series) -> float | None:
    """Return opening odds for the logged best outcome."""

    odds_column_by_outcome = {
        "home_win": "home_odds",
        "draw": "draw_odds",
        "away_win": "away_odds",
    }
    column = odds_column_by_outcome.get(str(row["best_outcome"]))

    if column is None:
        return None

    return _to_float(row[column])


def _selected_closing_odds(row: pd.Series) -> float | None:
    """Return closing odds for the logged best outcome when available."""

    odds_column_by_outcome = {
        "home_win": "closing_home_odds",
        "draw": "closing_draw_odds",
        "away_win": "closing_away_odds",
    }
    column = odds_column_by_outcome.get(str(row["best_outcome"]))

    if column is None:
        return None

    if not _has_value(row[column]):
        return None

    return _to_float(row[column])


def validate_prediction_ledger_for_report(ledger: pd.DataFrame) -> None:
    """Validate that a prediction ledger contains required reporting columns."""

    missing_columns = sorted(
        set(REQUIRED_LEDGER_REPORT_COLUMNS) - set(ledger.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction ledger missing required report columns: "
            f"{missing_columns}"
        )

    if ledger.empty:
        raise ValueError("Prediction ledger is empty.")


def summarize_prediction_ledger(ledger: pd.DataFrame) -> dict[str, float | int]:
    """Summarize settled prediction ledger performance."""

    validate_prediction_ledger_for_report(ledger)

    ledger = ledger.fillna("").copy()

    settled = ledger[
        ledger["final_outcome"].map(_has_value)
        & ledger["realized_return"].map(_has_value)
    ].copy()
    candidate_edges = ledger[ledger["decision"] == "candidate_edge"].copy()
    settled_candidate_edges = settled[settled["decision"] == "candidate_edge"].copy()

    total_predictions = len(ledger)
    settled_predictions = len(settled)
    candidate_edges_logged = len(candidate_edges)
    settled_candidate_count = len(settled_candidate_edges)

    if settled_candidate_count:
        realized_returns = settled_candidate_edges["realized_return"].map(_to_float)
        total_realized_return = float(realized_returns.sum())
        flat_stake_roi = total_realized_return / settled_candidate_count
        hit_rate = float(
            (
                settled_candidate_edges["best_outcome"]
                == settled_candidate_edges["final_outcome"]
            ).mean()
        )
        avg_expected_value = float(
            settled_candidate_edges["best_expected_value"].map(_to_float).mean()
        )
        avg_edge = float(settled_candidate_edges["best_edge"].map(_to_float).mean())
    else:
        total_realized_return = 0.0
        flat_stake_roi = 0.0
        hit_rate = 0.0
        avg_expected_value = 0.0
        avg_edge = 0.0

    closing_line_values: list[float] = []

    for _, row in settled_candidate_edges.iterrows():
        opening_odds = _selected_opening_odds(row)
        closing_odds = _selected_closing_odds(row)

        if opening_odds is not None and closing_odds is not None:
            closing_line_values.append(opening_odds - closing_odds)

    avg_closing_line_value = (
        float(sum(closing_line_values) / len(closing_line_values))
        if closing_line_values
        else 0.0
    )

    return {
        "total_predictions": total_predictions,
        "settled_predictions": settled_predictions,
        "candidate_edges_logged": candidate_edges_logged,
        "settled_candidate_edges": settled_candidate_count,
        "hit_rate": hit_rate,
        "total_realized_return": total_realized_return,
        "flat_stake_roi": flat_stake_roi,
        "avg_expected_value": avg_expected_value,
        "avg_edge": avg_edge,
        "avg_closing_line_value": avg_closing_line_value,
    }


def render_prediction_ledger_report(
    ledger: pd.DataFrame,
    title: str = "Prediction Ledger Performance Report",
) -> str:
    """Render a Markdown performance report from the prediction ledger."""

    validate_prediction_ledger_for_report(ledger)

    ledger = ledger.fillna("").copy()
    summary = summarize_prediction_ledger(ledger)

    settled = ledger[
        ledger["final_outcome"].map(_has_value)
        & ledger["realized_return"].map(_has_value)
    ].copy()

    settled_display = settled.tail(10)

    table_rows = [
        "| Prediction ID | Match | Decision | Pick | Final | Return | EV | Edge |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]

    for row in settled_display.itertuples(index=False):
        prediction_id = str(row.prediction_id)[:8]
        match = f"{row.home_team} vs {row.away_team}"
        table_rows.append(
            "| "
            f"{prediction_id} | "
            f"{match} | "
            f"{row.decision} | "
            f"{row.best_outcome} | "
            f"{row.final_outcome} | "
            f"{_format_money_return(_to_float(row.realized_return))} | "
            f"{_format_money_return(_to_float(row.best_expected_value))} | "
            f"{_format_probability(_to_float(row.best_edge))} |"
        )

    if settled.empty:
        settled_section = "No settled predictions yet."
    else:
        settled_section = "\n".join(table_rows)

    return f"""# {title}

## Purpose

This report summarizes the prediction ledger after forecasts have been logged
and, where available, settled with final match results.

The ledger is designed to support auditability, calibration review, expected
value tracking, and future betting-style evaluation.

## Summary Metrics

| Metric | Value |
|---|---:|
| Total logged predictions | {summary["total_predictions"]} |
| Settled predictions | {summary["settled_predictions"]} |
| Candidate edges logged | {summary["candidate_edges_logged"]} |
| Settled candidate edges | {summary["settled_candidate_edges"]} |
| Hit rate on settled candidate edges | {_format_probability(float(summary["hit_rate"]))} |
| Total flat-stake return | {_format_money_return(float(summary["total_realized_return"]))} |
| Flat-stake ROI | {_format_probability(float(summary["flat_stake_roi"]))} |
| Average expected value | {_format_money_return(float(summary["avg_expected_value"]))} |
| Average model edge | {_format_probability(float(summary["avg_edge"]))} |
| Average closing-line movement | {_format_money_return(float(summary["avg_closing_line_value"]))} |

## Recent Settled Predictions

{settled_section}

## Interpretation

The performance report treats rows with `decision = candidate_edge` as the
forecasting agent's candidate betting-style signals.

Rows with `decision = no_edge` are still useful for calibration and monitoring,
but they should not be counted as active betting decisions unless a strategy
explicitly says otherwise.

## Caveats

- The committed demo ledger is not evidence of real betting profitability.
- The current sample dataset is intentionally small.
- Real evaluation requires many timestamped, out-of-sample predictions.
- Closing-line value should be interpreted over large samples, not individual
  examples.
- This report is an audit and evaluation artifact, not betting advice.
"""


def save_prediction_ledger_report(
    ledger_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load a prediction ledger CSV and save a Markdown performance report."""

    ledger = pd.read_csv(ledger_path, keep_default_na=False, dtype=object)
    report = render_prediction_ledger_report(ledger)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report)

    return destination
