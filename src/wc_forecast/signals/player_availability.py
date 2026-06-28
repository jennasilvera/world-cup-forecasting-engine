from __future__ import annotations

from pathlib import Path

import pandas as pd

STATUS_WEIGHTS: dict[str, float] = {
    "available": 1.0,
    "fit": 1.0,
    "probable": 0.85,
    "questionable": 0.50,
    "doubtful": 0.25,
    "limited": 0.65,
    "out": 0.0,
    "injured": 0.0,
    "suspended": 0.0,
    "unavailable": 0.0,
}

REQUIRED_PLAYER_AVAILABILITY_COLUMNS = {
    "team",
    "player_name",
    "status",
    "importance_rating",
    "expected_minutes",
}


def normalize_availability_status(status: object) -> str:
    """Normalize player availability status text."""

    return str(status).strip().lower().replace(" ", "_")


def availability_status_weight(status: object) -> float:
    """Map a player availability status into an availability weight."""

    normalized = normalize_availability_status(status)
    return STATUS_WEIGHTS.get(normalized, 0.50)


def validate_player_availability_frame(frame: pd.DataFrame) -> None:
    """Validate required player availability columns."""

    missing = REQUIRED_PLAYER_AVAILABILITY_COLUMNS - set(frame.columns)

    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing player availability columns: {missing_columns}")


def build_team_availability_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level availability into team-level signal features."""

    validate_player_availability_frame(frame)

    availability = frame.copy()
    availability["status_normalized"] = availability["status"].map(
        normalize_availability_status
    )
    availability["status_weight"] = availability["status"].map(
        availability_status_weight
    )

    availability["importance_rating"] = pd.to_numeric(
        availability["importance_rating"],
        errors="coerce",
    ).fillna(0.0)
    availability["expected_minutes"] = pd.to_numeric(
        availability["expected_minutes"],
        errors="coerce",
    ).fillna(0.0)

    availability["importance_rating"] = availability["importance_rating"].clip(
        lower=0.0
    )
    availability["expected_minutes"] = availability["expected_minutes"].clip(
        lower=0.0,
        upper=120.0,
    )

    availability["baseline_player_value"] = (
        availability["importance_rating"] * availability["expected_minutes"] / 90.0
    )
    availability["available_player_value"] = (
        availability["baseline_player_value"] * availability["status_weight"]
    )
    availability["unavailable_player_value"] = (
        availability["baseline_player_value"] - availability["available_player_value"]
    )
    availability["available_expected_minutes"] = (
        availability["expected_minutes"] * availability["status_weight"]
    )
    availability["is_available"] = availability["status_weight"] >= 0.75
    availability["is_unavailable"] = availability["status_weight"] <= 0.05

    grouped = availability.groupby("team", dropna=False)

    features = grouped.agg(
        player_count=("player_name", "count"),
        available_player_count=("is_available", "sum"),
        unavailable_player_count=("is_unavailable", "sum"),
        expected_minutes_total=("expected_minutes", "sum"),
        expected_minutes_available=("available_expected_minutes", "sum"),
        baseline_player_value=("baseline_player_value", "sum"),
        available_player_value=("available_player_value", "sum"),
        unavailable_player_value=("unavailable_player_value", "sum"),
    ).reset_index()

    denominator = features["baseline_player_value"].replace(0.0, pd.NA)

    features["availability_pct"] = (
        features["available_player_value"].div(denominator).fillna(0.0)
    )
    features["unavailable_value_pct"] = (
        features["unavailable_player_value"].div(denominator).fillna(0.0)
    )

    return features.sort_values("team").reset_index(drop=True)


def build_fixture_availability_features(
    fixtures: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach home/away availability features to fixture rows."""

    required_fixture_columns = {"home_team", "away_team"}
    missing = required_fixture_columns - set(fixtures.columns)

    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing fixture columns: {missing_columns}")

    home_features = team_features.add_prefix("home_")
    away_features = team_features.add_prefix("away_")

    result = fixtures.merge(
        home_features,
        left_on="home_team",
        right_on="home_team",
        how="left",
    ).merge(
        away_features,
        left_on="away_team",
        right_on="away_team",
        how="left",
    )

    result["home_availability_pct"] = result["home_availability_pct"].fillna(1.0)
    result["away_availability_pct"] = result["away_availability_pct"].fillna(1.0)
    result["availability_pct_diff"] = (
        result["home_availability_pct"] - result["away_availability_pct"]
    )

    result["home_unavailable_value_pct"] = result[
        "home_unavailable_value_pct"
    ].fillna(0.0)
    result["away_unavailable_value_pct"] = result[
        "away_unavailable_value_pct"
    ].fillna(0.0)
    result["unavailable_value_pct_diff"] = (
        result["home_unavailable_value_pct"]
        - result["away_unavailable_value_pct"]
    )

    return result


def save_team_availability_features(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Read player availability CSV and write team-level features."""

    availability = pd.read_csv(input_path)
    features = build_team_availability_features(availability)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)

    return features
