from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

DEFAULT_DATABASE_PATH = Path("data/forecasting_engine.sqlite")


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS model_registry (
        model_id TEXT PRIMARY KEY,
        model_type TEXT NOT NULL,
        trained_at TEXT NOT NULL,
        train_cutoff_date TEXT NOT NULL,
        feature_version TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        artifact_path TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_store (
        feature_set_id TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        feature_date TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        feature_value REAL NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (
            feature_set_id,
            entity_type,
            entity_id,
            feature_date,
            feature_name
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_ledger (
        prediction_id TEXT PRIMARY KEY,
        model_id TEXT,
        fixture_id TEXT,
        prediction_timestamp TEXT NOT NULL,
        match_date TEXT NOT NULL,
        kickoff_at TEXT,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        predicted_outcome TEXT NOT NULL,
        predicted_winner TEXT NOT NULL,
        model_confidence REAL NOT NULL,
        prob_home_win REAL NOT NULL,
        prob_draw REAL NOT NULL,
        prob_away_win REAL NOT NULL,
        market_home_odds REAL,
        market_draw_odds REAL,
        market_away_odds REAL,
        closing_home_odds REAL,
        closing_draw_odds REAL,
        closing_away_odds REAL,
        realized_outcome TEXT,
        realized_winner TEXT,
        closing_line_value REAL,
        status TEXT NOT NULL DEFAULT 'open',
        FOREIGN KEY(model_id) REFERENCES model_registry(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_snapshots (
        market_snapshot_id TEXT PRIMARY KEY,
        fixture_id TEXT NOT NULL,
        snapshot_timestamp TEXT NOT NULL,
        bookmaker TEXT,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        source TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_availability (
        availability_id TEXT PRIMARY KEY,
        team TEXT NOT NULL,
        player_name TEXT NOT NULL,
        player_id TEXT,
        as_of TEXT NOT NULL,
        expected_status TEXT NOT NULL,
        availability_probability REAL NOT NULL,
        reason TEXT,
        source TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS injury_suspension_events (
        event_id TEXT PRIMARY KEY,
        team TEXT NOT NULL,
        player_name TEXT NOT NULL,
        player_id TEXT,
        event_type TEXT NOT NULL,
        status TEXT NOT NULL,
        start_date TEXT,
        expected_return_date TEXT,
        severity TEXT,
        source TEXT,
        reported_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scheduled_prediction_runs (
        run_id TEXT PRIMARY KEY,
        run_timestamp TEXT NOT NULL,
        as_of TEXT NOT NULL,
        fixtures_path TEXT NOT NULL,
        output_path TEXT NOT NULL,
        status TEXT NOT NULL,
        forecast_count INTEGER NOT NULL,
        message TEXT
    )
    """,
)


def connect_database(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite database connection with useful defaults."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    """Create all forecasting platform tables if they do not exist."""

    with connect_database(database_path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.commit()


def table_names(database_path: str | Path = DEFAULT_DATABASE_PATH) -> list[str]:
    """Return user-defined SQLite table names."""

    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

    return [str(row["name"]) for row in rows]


def insert_many(
    database_path: str | Path,
    table_name: str,
    rows: Iterable[dict[str, object]],
) -> int:
    """Insert dictionaries into a table and return inserted row count."""

    rows = list(rows)

    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)

    values = [tuple(row[column] for column in columns) for row in rows]

    with connect_database(database_path) as connection:
        connection.executemany(
            f"INSERT OR REPLACE INTO {table_name} ({column_sql}) VALUES ({placeholders})",
            values,
        )
        connection.commit()

    return len(rows)
