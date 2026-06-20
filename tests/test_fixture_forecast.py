from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.features.build_features import FEATURE_COLUMNS, build_match_features
from wc_forecast.forecasting.fixture_forecast import (
    build_fixture_forecast_features,
    build_ratings_before_cutoff,
    forecast_fixtures,
    validate_fixture_slate,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-11-20",
                    "2022-11-21",
                    "2022-11-21",
                    "2022-11-22",
                    "2022-11-22",
                    "2022-11-23",
                    "2022-11-24",
                    "2022-11-25",
                    "2022-12-03",
                    "2022-12-18",
                ]
            ),
            "home_team": [
                "Qatar",
                "England",
                "United States",
                "Argentina",
                "Mexico",
                "Spain",
                "Brazil",
                "Netherlands",
                "Netherlands",
                "Argentina",
            ],
            "away_team": [
                "Ecuador",
                "Iran",
                "Wales",
                "Saudi Arabia",
                "Poland",
                "Costa Rica",
                "Serbia",
                "Ecuador",
                "United States",
                "France",
            ],
            "home_score": [0, 6, 1, 1, 0, 7, 2, 1, 3, 3],
            "away_score": [2, 2, 1, 2, 0, 0, 0, 1, 1, 3],
            "tournament": ["FIFA World Cup"] * 10,
            "city": ["Doha"] * 10,
            "country": ["Qatar"] * 10,
            "neutral": [False, True, True, True, True, True, True, True, True, True],
            "outcome": [
                "away_win",
                "home_win",
                "draw",
                "away_win",
                "draw",
                "home_win",
                "home_win",
                "draw",
                "home_win",
                "draw",
            ],
        }
    )


def _sample_features() -> pd.DataFrame:
    return build_match_features(_sample_results())


def _sample_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["Argentina", "France", "Brazil", "Spain"],
            "elo_rating": [2043.4, 1997.4, 1957.9, 2026.1],
        }
    )


def _sample_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-06-20", "2026-06-21"],
            "home_team": ["Argentina", "Brazil"],
            "away_team": ["France", "Spain"],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "neutral": [True, "True"],
        }
    )


def test_validate_fixture_slate_rejects_missing_columns() -> None:
    fixtures = _sample_fixtures().drop(columns=["neutral"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_fixture_slate(fixtures)


def test_build_fixture_forecast_features_creates_model_columns() -> None:
    fixture_features = build_fixture_forecast_features(
        fixtures=_sample_fixtures(),
        ratings=_sample_ratings(),
    )

    assert set(FEATURE_COLUMNS).issubset(fixture_features.columns)
    assert fixture_features[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert fixture_features.loc[0, "home_elo_rating"] == pytest.approx(2043.4)


def test_forecast_fixtures_outputs_probabilities() -> None:
    forecasts = forecast_fixtures(
        fixtures=_sample_fixtures(),
        features=_sample_features(),
        ratings=_sample_ratings(),
        train_cutoff_date="2022-11-24",
    )

    assert len(forecasts) == 2
    assert {
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "predicted_winner",
        "model_confidence",
    }.issubset(forecasts.columns)

    probability_sums = forecasts[
        ["prob_home_win", "prob_draw", "prob_away_win"]
    ].sum(axis=1)

    assert probability_sums.round(8).eq(1.0).all()


def test_build_fixture_forecast_features_resolves_team_aliases() -> None:
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-20"],
            "home_team": ["Ecuador"],
            "away_team": ["Curacao"],
            "tournament": ["FIFA World Cup"],
            "neutral": [True],
        }
    )
    ratings = pd.DataFrame(
        {
            "team": ["Ecuador", "Curaçao"],
            "elo_rating": [1860.4, 1701.2],
        }
    )

    fixture_features = build_fixture_forecast_features(
        fixtures=fixtures,
        ratings=ratings,
    )

    assert fixture_features.loc[0, "away_elo_rating"] == pytest.approx(1701.2)
    assert fixture_features.loc[0, "away_rating_source"] == "alias_lookup"
    assert fixture_features.loc[0, "rating_warning"] == ""


def test_build_fixture_forecast_features_warns_on_unknown_team_rating() -> None:
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-20"],
            "home_team": ["Ecuador"],
            "away_team": ["Atlantis"],
            "tournament": ["FIFA World Cup"],
            "neutral": [True],
        }
    )
    ratings = pd.DataFrame(
        {
            "team": ["Ecuador"],
            "elo_rating": [1860.4],
        }
    )

    fixture_features = build_fixture_forecast_features(
        fixtures=fixtures,
        ratings=ratings,
    )

    assert fixture_features.loc[0, "away_elo_rating"] == pytest.approx(1500.0)
    assert fixture_features.loc[0, "away_rating_source"] == "fallback_1500"
    assert fixture_features.loc[0, "rating_warning"] == "fallback_rating_used:Atlantis"


def test_build_ratings_before_cutoff_excludes_future_matches() -> None:
    results = _sample_results()
    ratings = build_ratings_before_cutoff(
        results=results,
        rating_cutoff_date="2022-11-24",
    )

    assert not ratings.empty
    assert "Brazil" not in set(ratings["team"])


def test_build_ratings_before_cutoff_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="No historical result rows"):
        build_ratings_before_cutoff(
            results=_sample_results(),
            rating_cutoff_date="1900-01-01",
        )
