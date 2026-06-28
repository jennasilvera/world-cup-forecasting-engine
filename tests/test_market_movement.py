from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.signals.market_movement import build_market_movement_table


def test_build_market_movement_table_tracks_implied_probability_move() -> None:
    snapshots = pd.DataFrame(
        {
            "fixture_id": ["ARG_BRA_2026", "ARG_BRA_2026"],
            "snapshot_timestamp": [
                "2026-06-19T12:00:00Z",
                "2026-06-20T12:00:00Z",
            ],
            "home_team": ["Argentina", "Argentina"],
            "away_team": ["Brazil", "Brazil"],
            "market_home_odds": [2.20, 2.00],
            "market_draw_odds": [3.40, 3.50],
            "market_away_odds": [3.70, 4.10],
        }
    )

    movement = build_market_movement_table(snapshots)
    row = movement.iloc[0]

    assert row["fixture_id"] == "ARG_BRA_2026"
    assert row["snapshot_count"] == 2
    assert row["home_win_implied_probability_move"] == pytest.approx(
        1 / 2.00 - 1 / 2.20
    )
    assert row["largest_market_move_outcome"] == "home_win"
    assert row["market_direction"] == "market_toward_home_win"


def test_build_market_movement_table_rejects_invalid_timestamps() -> None:
    snapshots = pd.DataFrame(
        {
            "fixture_id": ["ARG_BRA_2026"],
            "snapshot_timestamp": ["not-a-date"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "market_home_odds": [2.20],
            "market_draw_odds": [3.40],
            "market_away_odds": [3.70],
        }
    )

    with pytest.raises(ValueError, match="invalid timestamps"):
        build_market_movement_table(snapshots)
