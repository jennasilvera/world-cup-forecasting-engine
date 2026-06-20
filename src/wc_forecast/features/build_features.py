from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.data.ingest_results import load_historical_results
from wc_forecast.models.elo import EloModel, match_importance_weight

FEATURE_COLUMNS = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_home_minus_away",
    "abs_elo_diff",
    "elo_expected_home_score",
    "elo_expected_away_score",
    "is_neutral",
    "is_world_cup",
    "tournament_importance",
]

TARGET_COLUMN = "outcome"

RESULT_COLUMNS = [
    "home_score",
    "away_score",
    "goal_diff_home_minus_away",
]


def build_match_features(results: pd.DataFrame) -> pd.DataFrame:
    """Build a pre-match feature table by replaying matches chronologically.

    Feature columns only use information available before kickoff. Final scores
    and match outcomes are included separately as target/result columns.
    """

    model = EloModel()
    feature_rows: list[dict[str, object]] = []

    sorted_results = results.sort_values(["date", "home_team", "away_team"]).reset_index(
        drop=True
    )

    for row in sorted_results.itertuples(index=False):
        prediction = model.predict_match(
            home_team=row.home_team,
            away_team=row.away_team,
            neutral=bool(row.neutral),
        )

        home_score = int(row.home_score)
        away_score = int(row.away_score)
        tournament_name = str(row.tournament)
        is_world_cup = tournament_name.strip().lower() == "fifa world cup"

        feature_rows.append(
            {
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "tournament": tournament_name,
                "neutral": bool(row.neutral),
                "home_elo_pre": prediction.home_rating,
                "away_elo_pre": prediction.away_rating,
                "elo_diff_home_minus_away": prediction.home_rating
                - prediction.away_rating,
                "abs_elo_diff": abs(prediction.home_rating - prediction.away_rating),
                "elo_expected_home_score": prediction.expected_home_score,
                "elo_expected_away_score": prediction.expected_away_score,
                "is_neutral": int(bool(row.neutral)),
                "is_world_cup": int(is_world_cup),
                "tournament_importance": match_importance_weight(tournament_name),
                "home_score": home_score,
                "away_score": away_score,
                "goal_diff_home_minus_away": home_score - away_score,
                "outcome": row.outcome,
            }
        )

        model.update_match(
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=home_score,
            away_score=away_score,
            tournament=tournament_name,
            neutral=bool(row.neutral),
        )

    return pd.DataFrame(feature_rows)


def validate_feature_table(features: pd.DataFrame) -> None:
    """Validate that the feature table is structurally safe for modeling."""

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "tournament",
        "outcome",
        *FEATURE_COLUMNS,
        *RESULT_COLUMNS,
    }

    missing_columns = sorted(required_columns - set(features.columns))
    if missing_columns:
        raise ValueError(f"Feature table missing required columns: {missing_columns}")

    if features.empty:
        raise ValueError("Feature table is empty.")

    if features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature table contains missing values in model features.")

    valid_outcomes = {"home_win", "draw", "away_win"}
    invalid_outcomes = set(features["outcome"]) - valid_outcomes

    if invalid_outcomes:
        raise ValueError(f"Invalid match outcomes found: {sorted(invalid_outcomes)}")


def save_match_features(
    results_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load processed results, build features, validate them, and save to CSV."""

    results = load_historical_results(results_path)
    features = build_match_features(results)
    validate_feature_table(features)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)

    return destination
