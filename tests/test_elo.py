from __future__ import annotations

import pandas as pd

from wc_forecast.models.elo import (
    EloModel,
    actual_score,
    expected_score,
    margin_of_victory_multiplier,
    match_importance_weight,
)


def test_expected_score_equal_ratings_is_even() -> None:
    assert expected_score(1500, 1500) == 0.5


def test_actual_score() -> None:
    assert actual_score(2, 1) == 1.0
    assert actual_score(1, 1) == 0.5
    assert actual_score(0, 1) == 0.0


def test_match_importance_weight_world_cup_above_friendly() -> None:
    assert match_importance_weight("FIFA World Cup") > match_importance_weight("Friendly")


def test_margin_of_victory_multiplier_increases_for_large_win() -> None:
    one_goal = margin_of_victory_multiplier(2, 1, 0)
    three_goal = margin_of_victory_multiplier(4, 1, 0)

    assert three_goal > one_goal


def test_elo_model_updates_winner_up_and_loser_down() -> None:
    model = EloModel()

    update = model.update_match(
        home_team="Argentina",
        away_team="France",
        home_score=2,
        away_score=0,
        tournament="FIFA World Cup",
        neutral=True,
    )

    assert update.home_rating_after > update.home_rating_before
    assert update.away_rating_after < update.away_rating_before


def test_elo_model_fit_returns_match_history() -> None:
    results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-17", "2022-12-18"]),
            "home_team": ["Croatia", "Argentina"],
            "away_team": ["Morocco", "France"],
            "home_score": [2, 3],
            "away_score": [1, 3],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "city": ["Al Rayyan", "Lusail"],
            "country": ["Qatar", "Qatar"],
            "neutral": [True, True],
            "outcome": ["home_win", "draw"],
        }
    )

    model = EloModel()
    history = model.fit(results)
    ratings = model.ratings_table()

    assert len(history) == 2
    assert {"home_rating_before", "away_rating_before", "rating_change"}.issubset(
        history.columns
    )
    assert len(ratings) == 4
