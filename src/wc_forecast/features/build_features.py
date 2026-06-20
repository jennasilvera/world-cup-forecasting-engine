from __future__ import annotations

from collections import defaultdict, deque
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
    "home_form_5_points_per_match",
    "away_form_5_points_per_match",
    "home_form_5_goal_diff_per_match",
    "away_form_5_goal_diff_per_match",
    "home_form_5_goals_for_per_match",
    "away_form_5_goals_for_per_match",
    "home_form_5_goals_against_per_match",
    "away_form_5_goals_against_per_match",
    "home_form_10_points_per_match",
    "away_form_10_points_per_match",
    "home_form_10_goal_diff_per_match",
    "away_form_10_goal_diff_per_match",
    "home_form_10_goals_for_per_match",
    "away_form_10_goals_for_per_match",
    "home_form_10_goals_against_per_match",
    "away_form_10_goals_against_per_match",
]

TARGET_COLUMN = "outcome"

RESULT_COLUMNS = [
    "home_score",
    "away_score",
    "goal_diff_home_minus_away",
]


FORM_WINDOWS = (5, 10)
DEFAULT_POINTS_PER_MATCH = 1.0
DEFAULT_GOAL_DIFF_PER_MATCH = 0.0
DEFAULT_GOALS_FOR_PER_MATCH = 1.0
DEFAULT_GOALS_AGAINST_PER_MATCH = 1.0


class TeamFormTracker:
    """Track leakage-safe rolling team form before each match update."""

    def __init__(self, windows: tuple[int, ...] = FORM_WINDOWS) -> None:
        self.windows = windows
        self.max_window = max(windows)
        self._team_points: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_window)
        )
        self._team_goal_diffs: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_window)
        )
        self._team_goals_for: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_window)
        )
        self._team_goals_against: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_window)
        )

    def prefixed_features(self, team: str, prefix: str) -> dict[str, float]:
        """Return prefixed rolling-form features for a team."""

        team_summary = self.team_summary(team)

        features: dict[str, float] = {}

        for window in self.windows:
            features[f"{prefix}_form_{window}_points_per_match"] = team_summary[
                f"form_{window}_points_per_match"
            ]
            features[f"{prefix}_form_{window}_goal_diff_per_match"] = team_summary[
                f"form_{window}_goal_diff_per_match"
            ]
            features[f"{prefix}_form_{window}_goals_for_per_match"] = team_summary[
                f"form_{window}_goals_for_per_match"
            ]
            features[f"{prefix}_form_{window}_goals_against_per_match"] = team_summary[
                f"form_{window}_goals_against_per_match"
            ]

        return features

    def team_summary(self, team: str) -> dict[str, float]:
        """Return unprefixed rolling-form features for a team."""

        team_name = str(team)
        points_history = list(self._team_points[team_name])
        goal_diff_history = list(self._team_goal_diffs[team_name])
        goals_for_history = list(self._team_goals_for[team_name])
        goals_against_history = list(self._team_goals_against[team_name])

        summary: dict[str, float] = {}

        for window in self.windows:
            recent_points = points_history[-window:]
            recent_goal_diffs = goal_diff_history[-window:]
            recent_goals_for = goals_for_history[-window:]
            recent_goals_against = goals_against_history[-window:]

            summary[f"form_{window}_points_per_match"] = _average_or_default(
                recent_points,
                DEFAULT_POINTS_PER_MATCH,
            )
            summary[f"form_{window}_goal_diff_per_match"] = _average_or_default(
                recent_goal_diffs,
                DEFAULT_GOAL_DIFF_PER_MATCH,
            )
            summary[f"form_{window}_goals_for_per_match"] = _average_or_default(
                recent_goals_for,
                DEFAULT_GOALS_FOR_PER_MATCH,
            )
            summary[f"form_{window}_goals_against_per_match"] = _average_or_default(
                recent_goals_against,
                DEFAULT_GOALS_AGAINST_PER_MATCH,
            )

        return summary

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
    ) -> None:
        """Update rolling form after the match result is known."""

        if home_score > away_score:
            home_points = 3.0
            away_points = 0.0
        elif home_score < away_score:
            home_points = 0.0
            away_points = 3.0
        else:
            home_points = 1.0
            away_points = 1.0

        home_goal_diff = float(home_score - away_score)
        away_goal_diff = float(away_score - home_score)

        self._team_points[str(home_team)].append(home_points)
        self._team_points[str(away_team)].append(away_points)
        self._team_goal_diffs[str(home_team)].append(home_goal_diff)
        self._team_goal_diffs[str(away_team)].append(away_goal_diff)
        self._team_goals_for[str(home_team)].append(float(home_score))
        self._team_goals_for[str(away_team)].append(float(away_score))
        self._team_goals_against[str(home_team)].append(float(away_score))
        self._team_goals_against[str(away_team)].append(float(home_score))


def _average_or_default(values: list[float], default: float) -> float:
    """Return average of values or a neutral default if no history exists."""

    if not values:
        return default

    return float(sum(values) / len(values))


def feature_default_value(column: str) -> float:
    """Return neutral default value for a model feature column."""

    if column.endswith("_points_per_match"):
        return DEFAULT_POINTS_PER_MATCH

    if column.endswith("_goal_diff_per_match"):
        return DEFAULT_GOAL_DIFF_PER_MATCH

    if column.endswith("_goals_for_per_match"):
        return DEFAULT_GOALS_FOR_PER_MATCH

    if column.endswith("_goals_against_per_match"):
        return DEFAULT_GOALS_AGAINST_PER_MATCH

    return 0.0


def ensure_model_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Ensure all model feature columns exist and contain finite defaults."""

    completed = features.copy()

    for column in FEATURE_COLUMNS:
        if column not in completed.columns:
            completed[column] = feature_default_value(column)

    for column in FEATURE_COLUMNS:
        completed[column] = pd.to_numeric(completed[column], errors="coerce")
        completed[column] = completed[column].fillna(feature_default_value(column))

    return completed


def build_match_features(results: pd.DataFrame) -> pd.DataFrame:
    """Build a pre-match feature table by replaying matches chronologically.

    Feature columns only use information available before kickoff. Final scores
    and match outcomes are included separately as target/result columns.
    """

    model = EloModel()
    form_tracker = TeamFormTracker()
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
                **form_tracker.prefixed_features(str(row.home_team), "home"),
                **form_tracker.prefixed_features(str(row.away_team), "away"),
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
        form_tracker.update_match(
            home_team=str(row.home_team),
            away_team=str(row.away_team),
            home_score=home_score,
            away_score=away_score,
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
