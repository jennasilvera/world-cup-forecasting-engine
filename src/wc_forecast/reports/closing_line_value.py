from __future__ import annotations

from pathlib import Path

import pandas as pd

OUTCOME_ODDS_COLUMNS = {
    "home_win": ("market_home_odds", "closing_home_odds"),
    "draw": ("market_draw_odds", "closing_draw_odds"),
    "away_win": ("market_away_odds", "closing_away_odds"),
}


def decimal_implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""

    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")

    return 1.0 / odds


def calculate_selected_outcome_clv(row: pd.Series) -> float | None:
    """Calculate closing-line value for the predicted outcome.

    Positive values mean the selected outcome's market price moved in the forecast's
    favor from market snapshot to close.
    """

    predicted_outcome = str(row["predicted_outcome"])
    odds_columns = OUTCOME_ODDS_COLUMNS.get(predicted_outcome)

    if odds_columns is None:
        return None

    market_column, closing_column = odds_columns
    market_odds = row.get(market_column)
    closing_odds = row.get(closing_column)

    if pd.isna(market_odds) or pd.isna(closing_odds):
        return None

    market_odds = float(market_odds)
    closing_odds = float(closing_odds)

    if market_odds <= 1.0 or closing_odds <= 1.0:
        return None

    return (market_odds / closing_odds) - 1.0


def build_clv_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build a closing-line value analytics table."""

    required_columns = {
        "prediction_id",
        "match_date",
        "home_team",
        "away_team",
        "predicted_outcome",
        "predicted_winner",
        "model_confidence",
        "market_home_odds",
        "market_draw_odds",
        "market_away_odds",
        "closing_home_odds",
        "closing_draw_odds",
        "closing_away_odds",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(f"Prediction dataframe missing columns: {sorted(missing_columns)}")

    table = predictions.copy()
    table["closing_line_value"] = table.apply(calculate_selected_outcome_clv, axis=1)
    table["closing_line_value_pct"] = table["closing_line_value"] * 100.0

    table["market_implied_probability"] = table.apply(
        lambda row: _selected_implied_probability(row, market=True),
        axis=1,
    )
    table["closing_implied_probability"] = table.apply(
        lambda row: _selected_implied_probability(row, market=False),
        axis=1,
    )

    output_columns = [
        "prediction_id",
        "match_date",
        "home_team",
        "away_team",
        "predicted_outcome",
        "predicted_winner",
        "model_confidence",
        "market_implied_probability",
        "closing_implied_probability",
        "closing_line_value",
        "closing_line_value_pct",
    ]

    return table[output_columns]


def save_clv_table(
    predictions_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Save closing-line value analytics from a prediction ledger CSV."""

    predictions = pd.read_csv(predictions_path)
    clv = build_clv_table(predictions)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clv.to_csv(destination, index=False)

    return clv


def _selected_implied_probability(row: pd.Series, market: bool) -> float | None:
    predicted_outcome = str(row["predicted_outcome"])
    odds_columns = OUTCOME_ODDS_COLUMNS.get(predicted_outcome)

    if odds_columns is None:
        return None

    column = odds_columns[0] if market else odds_columns[1]
    odds = row.get(column)

    if pd.isna(odds):
        return None

    odds = float(odds)

    if odds <= 1.0:
        return None

    return decimal_implied_probability(odds)
