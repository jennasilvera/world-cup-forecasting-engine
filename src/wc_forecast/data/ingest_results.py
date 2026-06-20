from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_RESULT_COLUMNS = [
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

TEXT_COLUMNS = [
    "home_team",
    "away_team",
    "tournament",
    "city",
    "country",
]


def _parse_bool(value: object) -> bool:
    """Parse common CSV boolean values into Python booleans."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Invalid neutral-site value: {value!r}")


def validate_results_frame(results: pd.DataFrame) -> None:
    """Validate that a historical results dataframe is safe to use."""

    missing_columns = sorted(set(REQUIRED_RESULT_COLUMNS) - set(results.columns))
    if missing_columns:
        raise ValueError(f"Missing required result columns: {missing_columns}")

    for column in REQUIRED_RESULT_COLUMNS:
        if results[column].isna().any():
            raise ValueError(f"Column contains missing values: {column}")

    for column in TEXT_COLUMNS:
        cleaned = results[column].astype(str).str.strip()
        if cleaned.eq("").any():
            raise ValueError(f"Column contains blank text values: {column}")

    home_teams = results["home_team"].astype(str).str.strip()
    away_teams = results["away_team"].astype(str).str.strip()

    if (home_teams == away_teams).any():
        raise ValueError("A team cannot play itself.")

    for score_column in ["home_score", "away_score"]:
        scores = pd.to_numeric(results[score_column], errors="raise")

        if (scores < 0).any():
            raise ValueError(f"Scores cannot be negative: {score_column}")

        if (scores % 1 != 0).any():
            raise ValueError(f"Scores must be whole numbers: {score_column}")


def load_historical_results(input_path: str | Path) -> pd.DataFrame:
    """Load, validate, clean, and sort historical international football results."""

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Historical results file not found: {path}")

    results = pd.read_csv(path)
    validate_results_frame(results)

    cleaned = results.copy()

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="raise")

    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    cleaned["home_score"] = pd.to_numeric(cleaned["home_score"], errors="raise").astype(int)
    cleaned["away_score"] = pd.to_numeric(cleaned["away_score"], errors="raise").astype(int)
    cleaned["neutral"] = cleaned["neutral"].map(_parse_bool)

    cleaned["outcome"] = "draw"
    cleaned.loc[cleaned["home_score"] > cleaned["away_score"], "outcome"] = "home_win"
    cleaned.loc[cleaned["home_score"] < cleaned["away_score"], "outcome"] = "away_win"

    cleaned = cleaned.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    return cleaned


def save_processed_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Load raw results and write a cleaned processed CSV."""

    processed = load_historical_results(input_path)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination, index=False)

    return destination
