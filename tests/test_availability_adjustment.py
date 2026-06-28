from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.signals.availability_adjustment import (
    apply_availability_adjustment,
)


def test_apply_availability_adjustment_favors_healthier_home_team() -> None:
    forecasts = pd.DataFrame(
        {
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "prob_home_win": [0.45],
            "prob_draw": [0.25],
            "prob_away_win": [0.30],
        }
    )
    availability = pd.DataFrame(
        {
            "team": ["Argentina", "Brazil"],
            "availability_pct": [1.00, 0.70],
            "unavailable_value_pct": [0.00, 0.30],
        }
    )

    adjusted = apply_availability_adjustment(forecasts, availability)
    row = adjusted.iloc[0]

    assert row["adjusted_prob_home_win"] > row["prob_home_win"]
    assert row["adjusted_prob_away_win"] < row["prob_away_win"]
    assert row["availability_adjustment_applied"]


def test_apply_availability_adjustment_preserves_probability_sum() -> None:
    forecasts = pd.DataFrame(
        {
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "prob_home_win": [0.40],
            "prob_draw": [0.30],
            "prob_away_win": [0.30],
        }
    )
    availability = pd.DataFrame(
        {
            "team": ["Argentina", "Brazil"],
            "availability_pct": [0.80, 0.90],
            "unavailable_value_pct": [0.20, 0.10],
        }
    )

    adjusted = apply_availability_adjustment(forecasts, availability)
    row = adjusted.iloc[0]

    total = (
        row["adjusted_prob_home_win"]
        + row["adjusted_prob_draw"]
        + row["adjusted_prob_away_win"]
    )

    assert total == pytest.approx(1.0)


def test_apply_availability_adjustment_defaults_missing_team_to_full_health() -> None:
    forecasts = pd.DataFrame(
        {
            "home_team": ["Argentina"],
            "away_team": ["Unknown Team"],
            "prob_home_win": [0.40],
            "prob_draw": [0.30],
            "prob_away_win": [0.30],
        }
    )
    availability = pd.DataFrame(
        {
            "team": ["Argentina"],
            "availability_pct": [0.80],
            "unavailable_value_pct": [0.20],
        }
    )

    adjusted = apply_availability_adjustment(forecasts, availability)

    assert adjusted.iloc[0]["away_availability_pct"] == 1.0
