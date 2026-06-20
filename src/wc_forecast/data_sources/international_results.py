from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_RESULTS_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

NORMALIZED_RESULTS_COLUMNS = SOURCE_RESULTS_COLUMNS + ["outcome"]

SUPPORTED_SOURCE_COLUMNS = set(SOURCE_RESULTS_COLUMNS)


def validate_international_results_source(results: pd.DataFrame) -> None:
    """Validate a real-world international-results CSV before normalization."""

    missing_columns = sorted(SUPPORTED_SOURCE_COLUMNS - set(results.columns))

    if missing_columns:
        raise ValueError(
            f"International results source missing columns: {missing_columns}"
        )

    if results.empty:
        raise ValueError("International results source is empty.")

    if results[["home_team", "away_team"]].isna().any().any():
        raise ValueError("International results source contains missing team names.")

    if results[["home_score", "away_score"]].isna().any().any():
        raise ValueError("International results source contains missing scores.")


def normalize_international_results(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize international match results into the engine schema."""

    validate_international_results_source(results)

    normalized = results[SOURCE_RESULTS_COLUMNS].copy()

    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date
    normalized["date"] = normalized["date"].astype(str)

    normalized["home_team"] = normalized["home_team"].astype(str).str.strip()
    normalized["away_team"] = normalized["away_team"].astype(str).str.strip()
    normalized["tournament"] = normalized["tournament"].astype(str).str.strip()
    normalized["city"] = normalized["city"].astype(str).str.strip()
    normalized["country"] = normalized["country"].astype(str).str.strip()

    normalized["home_score"] = normalized["home_score"].astype(int)
    normalized["away_score"] = normalized["away_score"].astype(int)

    normalized["neutral"] = normalized["neutral"].map(_coerce_bool)

    if (normalized["home_team"] == normalized["away_team"]).any():
        raise ValueError("International results source contains identical teams.")

    if (normalized[["home_score", "away_score"]] < 0).any().any():
        raise ValueError("International results source contains negative scores.")

    normalized["outcome"] = normalized.apply(_match_outcome, axis=1)

    return normalized[NORMALIZED_RESULTS_COLUMNS].sort_values("date").reset_index(drop=True)


def save_normalized_international_results(
    source_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load, normalize, and save real-world international results."""

    source = pd.read_csv(source_path)
    normalized = normalize_international_results(source)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(destination, index=False)

    return destination


def _match_outcome(row: pd.Series) -> str:
    """Return the match outcome from the home team's perspective."""

    if row["home_score"] > row["away_score"]:
        return "home_win"

    if row["home_score"] < row["away_score"]:
        return "away_win"

    return "draw"


def _coerce_bool(value: object) -> bool:
    """Convert common CSV bool-like values to bool."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Cannot convert value to bool: {value}")
