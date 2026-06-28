from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.signals.player_availability import (
    availability_status_weight,
    build_fixture_availability_features,
    build_team_availability_features,
)


def test_availability_status_weight_maps_known_statuses() -> None:
    assert availability_status_weight("available") == 1.0
    assert availability_status_weight("probable") == 0.85
    assert availability_status_weight("questionable") == 0.50
    assert availability_status_weight("suspended") == 0.0


def test_build_team_availability_features() -> None:
    availability = pd.DataFrame(
        {
            "team": ["Argentina", "Argentina", "Brazil"],
            "player_name": ["Player A", "Player B", "Player C"],
            "status": ["available", "out", "questionable"],
            "importance_rating": [90, 80, 75],
            "expected_minutes": [90, 90, 60],
        }
    )

    features = build_team_availability_features(availability)

    argentina = features.loc[features["team"] == "Argentina"].iloc[0]
    brazil = features.loc[features["team"] == "Brazil"].iloc[0]

    assert argentina["player_count"] == 2
    assert argentina["available_player_count"] == 1
    assert argentina["unavailable_player_count"] == 1
    assert argentina["availability_pct"] < 1.0
    assert brazil["availability_pct"] == 0.5


def test_build_fixture_availability_features() -> None:
    fixtures = pd.DataFrame(
        {
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
        }
    )
    team_features = pd.DataFrame(
        {
            "team": ["Argentina", "Brazil"],
            "availability_pct": [0.90, 0.70],
            "unavailable_value_pct": [0.10, 0.30],
        }
    )

    fixture_features = build_fixture_availability_features(
        fixtures,
        team_features,
    )

    row = fixture_features.iloc[0]

    assert row["home_availability_pct"] == pytest.approx(0.90)
    assert row["away_availability_pct"] == pytest.approx(0.70)
    assert row["availability_pct_diff"] == pytest.approx(0.20)
