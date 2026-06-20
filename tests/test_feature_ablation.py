from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.features.build_features import build_match_features
from wc_forecast.validation.feature_ablation import (
    FEATURE_SET_DEFINITIONS,
    neutralize_inactive_features,
    run_feature_ablation,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2019-01-01",
                    "2019-02-01",
                    "2019-03-01",
                    "2019-04-01",
                    "2019-05-01",
                    "2019-06-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                ]
            ),
            "home_team": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "away_team": ["B", "C", "A", "C", "A", "B", "B", "C", "A", "C"],
            "home_score": [2, 1, 1, 0, 3, 1, 2, 1, 0, 2],
            "away_score": [0, 1, 2, 1, 0, 1, 1, 2, 0, 0],
            "tournament": ["Friendly"] * 10,
            "city": ["X"] * 10,
            "country": ["Y"] * 10,
            "neutral": [True] * 10,
            "outcome": [
                "home_win",
                "draw",
                "away_win",
                "away_win",
                "home_win",
                "draw",
                "home_win",
                "away_win",
                "draw",
                "home_win",
            ],
        }
    )


def test_neutralize_inactive_features_preserves_active_features() -> None:
    features = build_match_features(_sample_results())
    active_features = FEATURE_SET_DEFINITIONS["elo_only"]

    ablated = neutralize_inactive_features(
        features=features,
        active_features=active_features,
    )

    assert ablated.loc[0, "home_elo_pre"] == features.loc[0, "home_elo_pre"]
    assert ablated.loc[0, "is_world_cup"] == 0.0
    assert ablated.loc[0, "home_form_5_points_per_match"] == 1.0


def test_run_feature_ablation_returns_ranked_feature_sets() -> None:
    features = build_match_features(_sample_results())

    result = run_feature_ablation(
        features=features,
        feature_set_names=["elo_context", "all_features"],
        cutoff_dates=["2020-01-01"],
        evaluation_window_days=365,
        sample_weight_half_life_days=365.0,
        logistic_c=0.5,
    )

    assert len(result) == 2
    assert set(result["feature_set"]) == {"elo_context", "all_features"}
    assert result["mean_log_loss"].is_monotonic_increasing


def test_run_feature_ablation_rejects_unknown_feature_set() -> None:
    features = build_match_features(_sample_results())

    with pytest.raises(ValueError, match="Unsupported feature_set_name"):
        run_feature_ablation(
            features=features,
            feature_set_names=["not_a_feature_set"],
            cutoff_dates=["2020-01-01"],
        )
