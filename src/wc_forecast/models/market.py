from __future__ import annotations

from dataclasses import dataclass

from wc_forecast.models.classifier import OUTCOME_ORDER


@dataclass(frozen=True)
class MarketProbabilities:
    """Market-implied probabilities derived from decimal odds."""

    raw_home_win: float
    raw_draw: float
    raw_away_win: float
    fair_home_win: float
    fair_draw: float
    fair_away_win: float
    overround: float


@dataclass(frozen=True)
class MarketEdge:
    """Comparison between model probabilities and market-implied probabilities."""

    edge_home_win: float
    edge_draw: float
    edge_away_win: float
    expected_value_home_win: float
    expected_value_draw: float
    expected_value_away_win: float
    best_outcome: str
    best_edge: float
    best_expected_value: float
    decision: str


def validate_decimal_odds(decimal_odds: float) -> None:
    """Validate decimal odds."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds into raw implied probability."""

    validate_decimal_odds(decimal_odds)
    return 1.0 / decimal_odds


def calculate_market_probabilities(
    home_win_odds: float,
    draw_odds: float,
    away_win_odds: float,
) -> MarketProbabilities:
    """Calculate raw and de-vigged probabilities from decimal odds."""

    raw_home = implied_probability(home_win_odds)
    raw_draw = implied_probability(draw_odds)
    raw_away = implied_probability(away_win_odds)

    overround = raw_home + raw_draw + raw_away

    if overround <= 0.0:
        raise ValueError("Market overround must be positive.")

    return MarketProbabilities(
        raw_home_win=raw_home,
        raw_draw=raw_draw,
        raw_away_win=raw_away,
        fair_home_win=raw_home / overround,
        fair_draw=raw_draw / overround,
        fair_away_win=raw_away / overround,
        overround=overround,
    )


def expected_value(model_probability: float, decimal_odds: float) -> float:
    """Calculate expected value for a decimal-odds bet."""

    validate_decimal_odds(decimal_odds)

    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("Model probability must be between 0 and 1.")

    return model_probability * decimal_odds - 1.0


def calculate_market_edge(
    model_prob_home_win: float,
    model_prob_draw: float,
    model_prob_away_win: float,
    home_win_odds: float,
    draw_odds: float,
    away_win_odds: float,
    minimum_edge: float = 0.03,
    minimum_expected_value: float = 0.02,
) -> MarketEdge:
    """Compare model probabilities against de-vigged market probabilities."""

    model_probabilities = {
        "home_win": model_prob_home_win,
        "draw": model_prob_draw,
        "away_win": model_prob_away_win,
    }

    if any(value < 0.0 or value > 1.0 for value in model_probabilities.values()):
        raise ValueError("Model probabilities must be between 0 and 1.")

    probability_sum = sum(model_probabilities.values())

    if probability_sum <= 0.0:
        raise ValueError("Model probabilities must sum to a positive value.")

    normalized_model_probabilities = {
        outcome: value / probability_sum
        for outcome, value in model_probabilities.items()
    }

    market = calculate_market_probabilities(
        home_win_odds=home_win_odds,
        draw_odds=draw_odds,
        away_win_odds=away_win_odds,
    )

    fair_market_probabilities = {
        "home_win": market.fair_home_win,
        "draw": market.fair_draw,
        "away_win": market.fair_away_win,
    }

    decimal_odds = {
        "home_win": home_win_odds,
        "draw": draw_odds,
        "away_win": away_win_odds,
    }

    edges = {
        outcome: (
            normalized_model_probabilities[outcome]
            - fair_market_probabilities[outcome]
        )
        for outcome in OUTCOME_ORDER
    }

    expected_values = {
        outcome: expected_value(
            model_probability=normalized_model_probabilities[outcome],
            decimal_odds=decimal_odds[outcome],
        )
        for outcome in OUTCOME_ORDER
    }

    best_outcome = max(
        OUTCOME_ORDER,
        key=lambda outcome: expected_values[outcome],
    )
    best_edge = edges[best_outcome]
    best_expected_value = expected_values[best_outcome]

    if best_edge >= minimum_edge and best_expected_value >= minimum_expected_value:
        decision = "candidate_edge"
    else:
        decision = "no_edge"

    return MarketEdge(
        edge_home_win=edges["home_win"],
        edge_draw=edges["draw"],
        edge_away_win=edges["away_win"],
        expected_value_home_win=expected_values["home_win"],
        expected_value_draw=expected_values["draw"],
        expected_value_away_win=expected_values["away_win"],
        best_outcome=best_outcome,
        best_edge=best_edge,
        best_expected_value=best_expected_value,
        decision=decision,
    )
