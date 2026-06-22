from __future__ import annotations

import pandas as pd

from wc_forecast.data.ingest_fixtures import normalize_world_cup_fixtures
from wc_forecast.forecasting.fixture_forecast import filter_upcoming_fixtures


def test_normalize_world_cup_fixtures_preserves_kickoff_at() -> None:
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-22"],
            "kickoff_at": ["2026-06-22T19:00:00-04:00"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
        }
    )

    normalized = normalize_world_cup_fixtures(fixtures)

    assert normalized.loc[0, "date"].isoformat() == "2026-06-22"
    assert normalized.loc[0, "kickoff_at"] == "2026-06-22T23:00:00Z"


def test_filter_upcoming_fixtures_excludes_matches_after_kickoff() -> None:
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-22", "2026-06-22"],
            "kickoff_at": [
                "2026-06-22T14:00:00Z",
                "2026-06-22T18:00:00Z",
            ],
            "home_team": ["Argentina", "England"],
            "away_team": ["Brazil", "France"],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "neutral": [True, True],
            "status": ["Scheduled", "Scheduled"],
        }
    )

    upcoming = filter_upcoming_fixtures(
        fixtures,
        as_of="2026-06-22T15:00:00Z",
    )

    assert upcoming["home_team"].tolist() == ["England"]


def test_filter_upcoming_fixtures_keeps_date_only_same_day_matches() -> None:
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-21", "2026-06-22"],
            "home_team": ["Argentina", "England"],
            "away_team": ["Brazil", "France"],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "neutral": [True, True],
            "status": ["Scheduled", "Scheduled"],
        }
    )

    upcoming = filter_upcoming_fixtures(
        fixtures,
        as_of="2026-06-22T15:00:00Z",
    )

    assert upcoming["home_team"].tolist() == ["England"]
