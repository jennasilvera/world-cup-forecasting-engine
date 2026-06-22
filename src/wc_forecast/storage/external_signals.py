from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc_forecast.storage.database import DEFAULT_DATABASE_PATH, insert_many


def write_market_snapshots(
    market: pd.DataFrame,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Write market odds snapshots into the database."""

    required_columns = {
        "fixture_id",
        "snapshot_timestamp",
        "home_team",
        "away_team",
    }

    missing_columns = required_columns - set(market.columns)

    if missing_columns:
        raise ValueError(f"Market snapshot dataframe missing columns: {sorted(missing_columns)}")

    rows: list[dict[str, object]] = []

    for row in market.itertuples(index=False):
        row_dict = row._asdict()
        rows.append(
            {
                "market_snapshot_id": f"market_{uuid.uuid4().hex[:12]}",
                "fixture_id": str(row_dict["fixture_id"]),
                "snapshot_timestamp": str(row_dict["snapshot_timestamp"]),
                "bookmaker": row_dict.get("bookmaker"),
                "home_team": str(row_dict["home_team"]),
                "away_team": str(row_dict["away_team"]),
                "home_odds": _nullable_float(row_dict.get("home_odds")),
                "draw_odds": _nullable_float(row_dict.get("draw_odds")),
                "away_odds": _nullable_float(row_dict.get("away_odds")),
                "source": row_dict.get("source"),
            }
        )

    return insert_many(database_path, "market_snapshots", rows)


def write_player_availability(
    availability: pd.DataFrame,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Write player-level availability probabilities into the database."""

    required_columns = {
        "team",
        "player_name",
        "as_of",
        "expected_status",
        "availability_probability",
    }

    missing_columns = required_columns - set(availability.columns)

    if missing_columns:
        raise ValueError(f"Availability dataframe missing columns: {sorted(missing_columns)}")

    rows: list[dict[str, object]] = []

    for row in availability.itertuples(index=False):
        row_dict = row._asdict()
        rows.append(
            {
                "availability_id": f"availability_{uuid.uuid4().hex[:12]}",
                "team": str(row_dict["team"]),
                "player_name": str(row_dict["player_name"]),
                "player_id": row_dict.get("player_id"),
                "as_of": str(row_dict["as_of"]),
                "expected_status": str(row_dict["expected_status"]),
                "availability_probability": float(row_dict["availability_probability"]),
                "reason": row_dict.get("reason"),
                "source": row_dict.get("source"),
            }
        )

    return insert_many(database_path, "player_availability", rows)


def write_injury_suspension_events(
    events: pd.DataFrame,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Write injury and suspension events into the database."""

    required_columns = {
        "team",
        "player_name",
        "event_type",
        "status",
    }

    missing_columns = required_columns - set(events.columns)

    if missing_columns:
        raise ValueError(f"Injury/suspension dataframe missing columns: {sorted(missing_columns)}")

    reported_at = datetime.now(UTC).isoformat()

    rows: list[dict[str, object]] = []

    for row in events.itertuples(index=False):
        row_dict = row._asdict()
        rows.append(
            {
                "event_id": f"event_{uuid.uuid4().hex[:12]}",
                "team": str(row_dict["team"]),
                "player_name": str(row_dict["player_name"]),
                "player_id": row_dict.get("player_id"),
                "event_type": str(row_dict["event_type"]),
                "status": str(row_dict["status"]),
                "start_date": row_dict.get("start_date"),
                "expected_return_date": row_dict.get("expected_return_date"),
                "severity": row_dict.get("severity"),
                "source": row_dict.get("source"),
                "reported_at": str(row_dict.get("reported_at") or reported_at),
            }
        )

    return insert_many(database_path, "injury_suspension_events", rows)


def _nullable_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None

    return float(value)
