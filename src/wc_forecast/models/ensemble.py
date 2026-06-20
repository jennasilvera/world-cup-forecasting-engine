from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from wc_forecast.models.classifier import OUTCOME_ORDER


@dataclass(frozen=True)
class EnsembleForecast:
    """Combined forecast from multiple model probability layers."""

    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    predicted_outcome: str
    confidence: str
    entropy: float
    max_model_disagreement: float


def _validate_weight(weight: float) -> None:
    """Validate a model blending weight."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("logistic_weight must be between 0 and 1.")


def _normalize_probability_map(
    probabilities: Mapping[str, float],
    name: str,
) -> dict[str, float]:
    """Validate and normalize an outcome probability mapping."""

    missing_outcomes = sorted(set(OUTCOME_ORDER) - set(probabilities))

    if missing_outcomes:
        raise ValueError(f"{name} probabilities missing outcomes: {missing_outcomes}")

    normalized = {outcome: float(probabilities[outcome]) for outcome in OUTCOME_ORDER}

    if any(value < 0.0 for value in normalized.values()):
        raise ValueError(f"{name} probabilities cannot be negative.")

    total = sum(normalized.values())

    if total <= 0.0:
        raise ValueError(f"{name} probabilities must sum to a positive value.")

    return {outcome: value / total for outcome, value in normalized.items()}


def probability_entropy(probabilities: Mapping[str, float]) -> float:
    """Compute normalized entropy for a three-outcome forecast."""

    normalized = _normalize_probability_map(probabilities, name="input")
    entropy = -sum(
        probability * math.log(probability)
        for probability in normalized.values()
        if probability > 0.0
    )

    return float(entropy / math.log(len(OUTCOME_ORDER)))


def max_probability_disagreement(
    first_probabilities: Mapping[str, float],
    second_probabilities: Mapping[str, float],
) -> float:
    """Return the largest absolute probability difference between two layers."""

    first = _normalize_probability_map(first_probabilities, name="first")
    second = _normalize_probability_map(second_probabilities, name="second")

    return max(abs(first[outcome] - second[outcome]) for outcome in OUTCOME_ORDER)


def confidence_label(probabilities: Mapping[str, float]) -> str:
    """Assign a simple confidence label from probability concentration."""

    normalized = _normalize_probability_map(probabilities, name="input")
    strongest_probability = max(normalized.values())
    entropy = probability_entropy(normalized)

    if strongest_probability >= 0.60 and entropy <= 0.75:
        return "High"

    if strongest_probability >= 0.45:
        return "Medium"

    return "Low"


def weighted_average_probabilities(
    logistic_probabilities: Mapping[str, float],
    poisson_probabilities: Mapping[str, float],
    logistic_weight: float = 0.50,
) -> dict[str, float]:
    """Blend logistic and Poisson probabilities with a weighted average."""

    _validate_weight(logistic_weight)

    logistic = _normalize_probability_map(
        logistic_probabilities,
        name="logistic",
    )
    poisson = _normalize_probability_map(
        poisson_probabilities,
        name="poisson",
    )

    poisson_weight = 1.0 - logistic_weight

    blended = {
        outcome: logistic_weight * logistic[outcome]
        + poisson_weight * poisson[outcome]
        for outcome in OUTCOME_ORDER
    }

    return _normalize_probability_map(blended, name="ensemble")


def build_ensemble_forecast(
    logistic_probabilities: Mapping[str, float],
    poisson_probabilities: Mapping[str, float],
    logistic_weight: float = 0.50,
) -> EnsembleForecast:
    """Build an ensemble forecast from logistic and Poisson model layers."""

    blended = weighted_average_probabilities(
        logistic_probabilities=logistic_probabilities,
        poisson_probabilities=poisson_probabilities,
        logistic_weight=logistic_weight,
    )

    predicted_outcome = max(blended, key=blended.get)
    disagreement = max_probability_disagreement(
        logistic_probabilities,
        poisson_probabilities,
    )

    return EnsembleForecast(
        prob_home_win=blended["home_win"],
        prob_draw=blended["draw"],
        prob_away_win=blended["away_win"],
        predicted_outcome=predicted_outcome,
        confidence=confidence_label(blended),
        entropy=probability_entropy(blended),
        max_model_disagreement=disagreement,
    )


def combine_match_prediction(
    prediction: Mapping[str, object],
    logistic_weight: float = 0.50,
) -> EnsembleForecast:
    """Build an ensemble forecast from a match-prediction dictionary."""

    logistic_probabilities = {
        "home_win": float(prediction["logistic_prob_home_win"]),
        "draw": float(prediction["logistic_prob_draw"]),
        "away_win": float(prediction["logistic_prob_away_win"]),
    }

    poisson_probabilities = {
        "home_win": float(prediction["poisson_prob_home_win"]),
        "draw": float(prediction["poisson_prob_draw"]),
        "away_win": float(prediction["poisson_prob_away_win"]),
    }

    return build_ensemble_forecast(
        logistic_probabilities=logistic_probabilities,
        poisson_probabilities=poisson_probabilities,
        logistic_weight=logistic_weight,
    )
