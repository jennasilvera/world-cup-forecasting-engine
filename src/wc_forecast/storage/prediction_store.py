from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc_forecast.storage.database import DEFAULT_DATABASE_PATH, insert_many


def write_forecasts_to_prediction_ledger(
    forecasts: pd.DataFrame,
    model_id: str | None = None,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Write forecast rows into the database-backed prediction ledger."""

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "predicted_outcome",
        "predicted_winner",
        "model_confidence",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
    }

    missing_columns = required_columns - set(forecasts.columns)

    if missing_columns:
        raise ValueError(f"Forecast dataframe missing columns: {sorted(missing_columns)}")

    prediction_timestamp = datetime.now(UTC).isoformat()

    rows: list[dict[str, object]] = []

    for row in forecasts.itertuples(index=False):
        row_dict = row._asdict()
        home_team = str(row_dict["home_team"])
        away_team = str(row_dict["away_team"])
        match_date = str(pd.Timestamp(row_dict["date"]).date())
        kickoff_at = row_dict.get("kickoff_at")

        prediction_id = (
            f"pred_{match_date}_{home_team}_{away_team}_{uuid.uuid4().hex[:8]}"
            .replace(" ", "_")
            .replace("/", "_")
        )

        rows.append(
            {
                "prediction_id": prediction_id,
                "model_id": model_id,
                "fixture_id": f"{match_date}:{home_team}:{away_team}",
                "prediction_timestamp": prediction_timestamp,
                "match_date": match_date,
                "kickoff_at": "" if pd.isna(kickoff_at) else str(kickoff_at),
                "home_team": home_team,
                "away_team": away_team,
                "predicted_outcome": str(row_dict["predicted_outcome"]),
                "predicted_winner": str(row_dict["predicted_winner"]),
                "model_confidence": float(row_dict["model_confidence"]),
                "prob_home_win": float(row_dict["prob_home_win"]),
                "prob_draw": float(row_dict["prob_draw"]),
                "prob_away_win": float(row_dict["prob_away_win"]),
                "market_home_odds": None,
                "market_draw_odds": None,
                "market_away_odds": None,
                "closing_home_odds": None,
                "closing_draw_odds": None,
                "closing_away_odds": None,
                "realized_outcome": None,
                "realized_winner": None,
                "closing_line_value": None,
                "status": "open",
            }
        )

    return insert_many(database_path, "prediction_ledger", rows)


def read_prediction_ledger(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, object]]:
    """Read prediction ledger rows ordered by newest first."""

    from wc_forecast.storage.database import connect_database

    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM prediction_ledger
            ORDER BY prediction_timestamp DESC, match_date ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]
