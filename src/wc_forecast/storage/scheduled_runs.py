from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from wc_forecast.storage.database import DEFAULT_DATABASE_PATH, connect_database


def record_scheduled_prediction_run(
    as_of: str,
    fixtures_path: str,
    output_path: str,
    status: str,
    forecast_count: int,
    message: str | None = None,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> str:
    """Record an automated scheduled prediction run."""

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_timestamp = datetime.now(UTC).isoformat()

    with connect_database(database_path) as connection:
        connection.execute(
            """
            INSERT INTO scheduled_prediction_runs (
                run_id,
                run_timestamp,
                as_of,
                fixtures_path,
                output_path,
                status,
                forecast_count,
                message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_timestamp,
                as_of,
                fixtures_path,
                output_path,
                status,
                forecast_count,
                message,
            ),
        )
        connection.commit()

    return run_id


def list_scheduled_prediction_runs(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, object]]:
    """List scheduled prediction run audit rows."""

    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM scheduled_prediction_runs
            ORDER BY run_timestamp DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
