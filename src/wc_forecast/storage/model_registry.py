from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from wc_forecast.storage.database import DEFAULT_DATABASE_PATH, connect_database


def register_model(
    model_type: str,
    train_cutoff_date: str,
    feature_version: str,
    metrics: dict[str, float | int | str],
    artifact_path: str | None = None,
    notes: str | None = None,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> str:
    """Register a model training artifact and return its model id."""

    model_id = f"model_{uuid.uuid4().hex[:12]}"
    trained_at = datetime.now(UTC).isoformat()

    with connect_database(database_path) as connection:
        connection.execute(
            """
            INSERT INTO model_registry (
                model_id,
                model_type,
                trained_at,
                train_cutoff_date,
                feature_version,
                metrics_json,
                artifact_path,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                model_type,
                trained_at,
                train_cutoff_date,
                feature_version,
                json.dumps(metrics, sort_keys=True),
                artifact_path,
                notes,
            ),
        )
        connection.commit()

    return model_id


def list_models(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, object]]:
    """Return registered models ordered by newest first."""

    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM model_registry
            ORDER BY trained_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
