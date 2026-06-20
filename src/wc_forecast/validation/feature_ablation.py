from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.features.build_features import (
    FEATURE_COLUMNS,
    feature_default_value,
    validate_feature_table,
)
from wc_forecast.validation.rolling_backtest import (
    DEFAULT_ROLLING_CUTOFF_DATES,
    run_rolling_backtest,
    summarize_rolling_backtest,
)

ELO_FEATURES = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_home_minus_away",
    "abs_elo_diff",
    "elo_expected_home_score",
    "elo_expected_away_score",
]

CONTEXT_FEATURES = [
    "is_neutral",
    "is_world_cup",
    "tournament_importance",
]

POINTS_GOAL_DIFF_FORM_FEATURES = [
    "home_form_5_points_per_match",
    "away_form_5_points_per_match",
    "home_form_5_goal_diff_per_match",
    "away_form_5_goal_diff_per_match",
    "home_form_10_points_per_match",
    "away_form_10_points_per_match",
    "home_form_10_goal_diff_per_match",
    "away_form_10_goal_diff_per_match",
]

ATTACK_DEFENSE_FORM_FEATURES = [
    "home_form_5_goals_for_per_match",
    "away_form_5_goals_for_per_match",
    "home_form_5_goals_against_per_match",
    "away_form_5_goals_against_per_match",
    "home_form_10_goals_for_per_match",
    "away_form_10_goals_for_per_match",
    "home_form_10_goals_against_per_match",
    "away_form_10_goals_against_per_match",
]

FEATURE_SET_DEFINITIONS = {
    "elo_only": ELO_FEATURES,
    "elo_context": ELO_FEATURES + CONTEXT_FEATURES,
    "elo_context_points_form": (
        ELO_FEATURES + CONTEXT_FEATURES + POINTS_GOAL_DIFF_FORM_FEATURES
    ),
    "all_features": FEATURE_COLUMNS,
}

DEFAULT_FEATURE_SET_NAMES = list(FEATURE_SET_DEFINITIONS)


def run_feature_ablation(
    features: pd.DataFrame,
    feature_set_names: list[str] | None = None,
    cutoff_dates: list[str] | None = None,
    evaluation_window_days: int = 365,
    sample_weight_half_life_days: float | None = None,
    logistic_c: float = 1.0,
) -> pd.DataFrame:
    """Run rolling validation while neutralizing excluded feature groups."""

    validate_feature_table(features)

    active_feature_sets = feature_set_names or DEFAULT_FEATURE_SET_NAMES
    active_cutoffs = cutoff_dates or DEFAULT_ROLLING_CUTOFF_DATES

    rows: list[dict[str, object]] = []

    for feature_set_name in active_feature_sets:
        if feature_set_name not in FEATURE_SET_DEFINITIONS:
            raise ValueError(
                f"Unsupported feature_set_name: {feature_set_name}. "
                f"Expected one of: {sorted(FEATURE_SET_DEFINITIONS)}"
            )

        active_features = FEATURE_SET_DEFINITIONS[feature_set_name]
        ablated_features = neutralize_inactive_features(
            features=features,
            active_features=active_features,
        )

        rolling_results = run_rolling_backtest(
            features=ablated_features,
            cutoff_dates=active_cutoffs,
            model_types=["logistic"],
            evaluation_window_days=evaluation_window_days,
            sample_weight_half_life_days=sample_weight_half_life_days,
            logistic_c=logistic_c,
        )
        summary = summarize_rolling_backtest(rolling_results).iloc[0]

        rows.append(
            {
                "feature_set": feature_set_name,
                "active_feature_count": len(active_features),
                "folds": int(summary["folds"]),
                "total_test_rows": int(summary["total_test_rows"]),
                "mean_accuracy": float(summary["mean_accuracy"]),
                "mean_log_loss": float(summary["mean_log_loss"]),
                "mean_brier": float(summary["mean_brier"]),
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(["mean_log_loss", "mean_brier"]).reset_index(drop=True)

    return result


def neutralize_inactive_features(
    features: pd.DataFrame,
    active_features: list[str],
) -> pd.DataFrame:
    """Return features with inactive model columns set to neutral defaults."""

    active_feature_set = set(active_features)
    ablated = features.copy()

    for column in FEATURE_COLUMNS:
        if column not in active_feature_set:
            ablated[column] = feature_default_value(column)

    return ablated


def save_feature_ablation(
    features_path: str | Path,
    output_path: str | Path,
    feature_set_names: list[str] | None = None,
    cutoff_dates: list[str] | None = None,
    evaluation_window_days: int = 365,
    sample_weight_half_life_days: float | None = None,
    logistic_c: float = 1.0,
) -> pd.DataFrame:
    """Load features, run feature ablation, and save CSV."""

    features = pd.read_csv(features_path)
    result = run_feature_ablation(
        features=features,
        feature_set_names=feature_set_names,
        cutoff_dates=cutoff_dates,
        evaluation_window_days=evaluation_window_days,
        sample_weight_half_life_days=sample_weight_half_life_days,
        logistic_c=logistic_c,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)

    return result
