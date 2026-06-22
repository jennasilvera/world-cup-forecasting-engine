from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc_forecast.storage.database import DEFAULT_DATABASE_PATH, insert_many


def write_feature_store_frame(
    features: pd.DataFrame,
    entity_columns: list[str],
    feature_columns: list[str],
    feature_date_column: str = "date",
    entity_type: str = "team_match",
    feature_set_id: str | None = None,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> str:
    """Write a wide feature dataframe into a long-form feature store table."""

    missing_columns = (
        set(entity_columns)
        | set(feature_columns)
        | {feature_date_column}
    ) - set(features.columns)

    if missing_columns:
        raise ValueError(f"Feature dataframe missing columns: {sorted(missing_columns)}")

    feature_set_id = feature_set_id or f"features_{uuid.uuid4().hex[:12]}"
    created_at = datetime.now(UTC).isoformat()

    rows: list[dict[str, object]] = []

    for row in features.itertuples(index=False):
        row_dict = row._asdict()
        entity_id = "|".join(str(row_dict[column]) for column in entity_columns)
        feature_date = str(pd.Timestamp(row_dict[feature_date_column]).date())

        for feature_name in feature_columns:
            value = row_dict[feature_name]

            if pd.isna(value):
                continue

            rows.append(
                {
                    "feature_set_id": feature_set_id,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "feature_date": feature_date,
                    "feature_name": feature_name,
                    "feature_value": float(value),
                    "created_at": created_at,
                }
            )

    insert_many(database_path, "feature_store", rows)

    return feature_set_id
