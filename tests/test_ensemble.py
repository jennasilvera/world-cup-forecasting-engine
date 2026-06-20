from __future__ import annotations

import pytest

from wc_forecast.models.ensemble import (
    build_ensemble_forecast,
    combine_match_prediction,
    max_probability_disagreement,
    probability_entropy,
    weighted_average_probabilities,
)


def test_weighted_average_probabilities_sum_to_one() -> None:
    logistic = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
    poisson = {"home_win": 0.30, "draw": 0.30, "away_win": 0.40}

    blended = weighted_average_probabilities(logistic, poisson, logistic_weight=0.60)

    assert round(sum(blended.values()), 8) == 1.0
    assert blended["home_win"] > blended["away_win"]


def test_weighted_average_rejects_invalid_weight() -> None:
    logistic = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
    poisson = {"home_win": 0.30, "draw": 0.30, "away_win": 0.40}

    with pytest.raises(ValueError, match="logistic_weight"):
        weighted_average_probabilities(logistic, poisson, logistic_weight=1.50)


def test_probability_entropy_is_between_zero_and_one() -> None:
    probabilities = {"home_win": 0.34, "draw": 0.33, "away_win": 0.33}

    entropy = probability_entropy(probabilities)

    assert 0.0 <= entropy <= 1.0


def test_max_probability_disagreement_detects_layer_gap() -> None:
    logistic = {"home_win": 0.70, "draw": 0.20, "away_win": 0.10}
    poisson = {"home_win": 0.20, "draw": 0.20, "away_win": 0.60}

    disagreement = max_probability_disagreement(logistic, poisson)

    assert disagreement == 0.50


def test_build_ensemble_forecast_returns_forecast_object() -> None:
    logistic = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
    poisson = {"home_win": 0.30, "draw": 0.30, "away_win": 0.40}

    forecast = build_ensemble_forecast(logistic, poisson)

    assert round(forecast.prob_home_win + forecast.prob_draw + forecast.prob_away_win, 8) == 1.0
    assert forecast.predicted_outcome in {"home_win", "draw", "away_win"}
    assert forecast.confidence in {"High", "Medium", "Low"}


def test_combine_match_prediction_from_report_dictionary() -> None:
    prediction = {
        "logistic_prob_home_win": 0.50,
        "logistic_prob_draw": 0.25,
        "logistic_prob_away_win": 0.25,
        "poisson_prob_home_win": 0.30,
        "poisson_prob_draw": 0.30,
        "poisson_prob_away_win": 0.40,
    }

    forecast = combine_match_prediction(prediction)

    assert forecast.predicted_outcome == "home_win"
