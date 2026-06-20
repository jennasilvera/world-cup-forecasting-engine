from __future__ import annotations

import pytest

from wc_forecast.models.market import (
    calculate_market_edge,
    calculate_market_probabilities,
    expected_value,
    implied_probability,
)


def test_implied_probability_from_decimal_odds() -> None:
    assert implied_probability(2.0) == 0.5


def test_implied_probability_rejects_invalid_odds() -> None:
    with pytest.raises(ValueError, match="greater than 1.0"):
        implied_probability(1.0)


def test_calculate_market_probabilities_devigs_market() -> None:
    market = calculate_market_probabilities(
        home_win_odds=2.0,
        draw_odds=3.5,
        away_win_odds=4.0,
    )

    fair_sum = market.fair_home_win + market.fair_draw + market.fair_away_win

    assert market.overround > 1.0
    assert round(fair_sum, 8) == 1.0


def test_expected_value_calculation() -> None:
    assert expected_value(model_probability=0.55, decimal_odds=2.10) == pytest.approx(
        0.155
    )


def test_calculate_market_edge_returns_candidate_edge() -> None:
    edge = calculate_market_edge(
        model_prob_home_win=0.55,
        model_prob_draw=0.25,
        model_prob_away_win=0.20,
        home_win_odds=2.20,
        draw_odds=3.40,
        away_win_odds=3.50,
        minimum_edge=0.01,
        minimum_expected_value=0.01,
    )

    assert edge.best_outcome == "home_win"
    assert edge.best_expected_value > 0.0
    assert edge.decision == "candidate_edge"


def test_calculate_market_edge_can_return_no_edge() -> None:
    edge = calculate_market_edge(
        model_prob_home_win=0.34,
        model_prob_draw=0.33,
        model_prob_away_win=0.33,
        home_win_odds=2.20,
        draw_odds=3.40,
        away_win_odds=3.50,
        minimum_edge=0.20,
        minimum_expected_value=0.20,
    )

    assert edge.decision == "no_edge"
