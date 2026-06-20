from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.features.build_features import (
    FEATURE_COLUMNS,
    build_match_features,
    validate_feature_table,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-11-20",
                    "2022-11-21",
                    "2022-12-18",
                ]
            ),
            "home_team": ["Qatar", "England", "Argentina"],
            "away_team": ["Ecuador", "Iran", "France"],
            "home_score": [0, 6, 3],
            "away_score": [2, 2, 3],
            "tournament": [
                "FIFA World Cup",
                "FIFA World Cup",
                "FIFA World Cup",
            ],
            "city": ["Al Khor", "Al Rayyan", "Lusail"],
            "country": ["Qatar", "Qatar", "Qatar"],
            "neutral": [False, True, True],
            "outcome": ["away_win", "home_win", "draw"],
        }
    )


def test_build_match_features_creates_pre_match_columns() -> None:
    features = build_match_features(_sample_results())

    assert len(features) == 3

    for column in FEATURE_COLUMNS:
        assert column in features.columns

    assert "outcome" in features.columns
    assert "home_score" in features.columns
    assert "away_score" in features.columns


def test_first_match_uses_default_elo_ratings() -> None:
    features = build_match_features(_sample_results())

    first_row = features.iloc[0]

    assert first_row["home_elo_pre"] == 1500.0
    assert first_row["away_elo_pre"] == 1500.0
    assert first_row["elo_diff_home_minus_away"] == 0.0


def test_feature_table_has_no_missing_model_features() -> None:
    features = build_match_features(_sample_results())

    assert not features[FEATURE_COLUMNS].isna().any().any()


def test_validate_feature_table_rejects_invalid_outcome() -> None:
    features = build_match_features(_sample_results())
    features.loc[0, "outcome"] = "invalid"

    with pytest.raises(ValueError, match="Invalid match outcomes found"):
        validate_feature_table(features)


def test_validate_feature_table_rejects_missing_feature_column() -> None:
    features = build_match_features(_sample_results())
    features = features.drop(columns=["home_elo_pre"])

    with pytest.raises(ValueError, match="Feature table missing required columns"):
        validate_feature_table(features)
