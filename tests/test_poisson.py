from __future__ import annotations

import pandas as pd

from wc_forecast.models.poisson import (
    PoissonGoalsModel,
    poisson_pmf,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-11-20",
                    "2022-11-21",
                    "2022-11-22",
                    "2022-11-23",
                    "2022-11-24",
                    "2022-12-18",
                ]
            ),
            "home_team": [
                "Qatar",
                "England",
                "Argentina",
                "Spain",
                "Brazil",
                "Argentina",
            ],
            "away_team": [
                "Ecuador",
                "Iran",
                "Saudi Arabia",
                "Costa Rica",
                "Serbia",
                "France",
            ],
            "home_score": [0, 6, 1, 7, 2, 3],
            "away_score": [2, 2, 2, 0, 0, 3],
            "tournament": ["FIFA World Cup"] * 6,
            "city": [
                "Al Khor",
                "Al Rayyan",
                "Lusail",
                "Doha",
                "Lusail",
                "Lusail",
            ],
            "country": ["Qatar"] * 6,
            "neutral": [False, True, True, True, True, True],
            "outcome": [
                "away_win",
                "home_win",
                "away_win",
                "home_win",
                "home_win",
                "draw",
            ],
        }
    )


def test_poisson_pmf_returns_valid_probability() -> None:
    probability = poisson_pmf(goals=1, expected_goals=1.25)

    assert 0.0 < probability < 1.0


def test_poisson_model_fit_creates_team_strengths() -> None:
    model = PoissonGoalsModel()
    model.fit(_sample_results())

    assert "Argentina" in model.attack_strength
    assert "France" in model.defense_weakness
    assert model.global_goals_per_team_match > 0


def test_poisson_prediction_probabilities_sum_to_one() -> None:
    model = PoissonGoalsModel()
    model.fit(_sample_results())

    prediction = model.predict_match(
        home_team="Argentina",
        away_team="France",
        neutral=True,
    )

    probability_sum = (
        prediction.prob_home_win
        + prediction.prob_draw
        + prediction.prob_away_win
    )

    assert round(probability_sum, 8) == 1.0
    assert prediction.expected_home_goals > 0
    assert prediction.expected_away_goals > 0
    assert "-" in prediction.most_likely_score


def test_scoreline_probabilities_are_normalized() -> None:
    model = PoissonGoalsModel()
    scorelines = model.scoreline_probabilities(
        expected_home_goals=1.4,
        expected_away_goals=1.1,
    )

    assert round(float(scorelines["probability"].sum()), 8) == 1.0
    assert {"home_goals", "away_goals", "scoreline", "outcome", "probability"}.issubset(
        scorelines.columns
    )
