from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_FIXTURE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
]

OPTIONAL_FIXTURE_DEFAULTS = {
    "tournament": "FIFA World Cup",
    "neutral": True,
    "status": "Scheduled",
}


PROCESSED_FIXTURE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "tournament",
    "neutral",
    "status",
]


def normalize_world_cup_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a raw World Cup fixture schedule."""

    missing_columns = set(REQUIRED_FIXTURE_COLUMNS) - set(fixtures.columns)

    if missing_columns:
        raise ValueError(
            f"Fixture schedule missing required columns: {sorted(missing_columns)}"
        )

    normalized = fixtures.copy()

    for column, default_value in OPTIONAL_FIXTURE_DEFAULTS.items():
        if column not in normalized.columns:
            normalized[column] = default_value

    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date

    normalized["home_team"] = normalized["home_team"].astype(str).str.strip()
    normalized["away_team"] = normalized["away_team"].astype(str).str.strip()
    normalized["tournament"] = normalized["tournament"].astype(str).str.strip()
    normalized["status"] = normalized["status"].astype(str).str.strip()

    normalized["neutral"] = normalized["neutral"].map(_parse_bool)

    blank_team_mask = (
        normalized["home_team"].eq("")
        | normalized["away_team"].eq("")
        | normalized["home_team"].str.lower().eq("nan")
        | normalized["away_team"].str.lower().eq("nan")
    )

    if blank_team_mask.any():
        raise ValueError("Fixture schedule contains blank home_team or away_team values.")

    normalized = normalized[PROCESSED_FIXTURE_COLUMNS]
    normalized = normalized.sort_values(["date", "home_team", "away_team"]).reset_index(
        drop=True
    )

    return normalized


def save_world_cup_fixtures(
    source_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Load, normalize, and save a World Cup fixture schedule."""

    source = Path(source_path)
    destination = Path(output_path)

    fixtures = pd.read_csv(source)
    normalized = normalize_world_cup_fixtures(fixtures)

    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(destination, index=False)

    return normalized


def _parse_bool(value: object) -> bool:
    """Parse common CSV boolean representations."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Could not parse boolean value: {value!r}")
