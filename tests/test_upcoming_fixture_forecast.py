from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.forecasting.fixture_forecast import filter_upcoming_fixtures


def test_filter_upcoming_fixtures_excludes_past_completed_and_tbd() -> None:
    fixtures = pd.DataFrame(
        {
            "date": [
                "2026-06-20",
                "2026-06-22",
                "2026-06-23",
                "2026-06-24",
            ],
            "home_team": ["Past Team", "Argentina", "TBD", "France"],
            "away_team": ["Old Team", "Austria", "Brazil", "Iraq"],
            "tournament": ["FIFA World Cup"] * 4,
            "neutral": [True] * 4,
            "status": ["Scheduled", "Scheduled", "Scheduled", "Completed"],
        }
    )

    result = filter_upcoming_fixtures(
        fixtures=fixtures,
        from_date="2026-06-21",
    )

    assert len(result) == 1
    assert result.loc[0, "home_team"] == "Argentina"
    assert result.loc[0, "away_team"] == "Austria"


def test_filter_upcoming_fixtures_can_include_tbd() -> None:
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-23"],
            "home_team": ["TBD"],
            "away_team": ["Brazil"],
        }
    )

    result = filter_upcoming_fixtures(
        fixtures=fixtures,
        from_date="2026-06-21",
        include_tbd=True,
    )

    assert len(result) == 1


def test_filter_upcoming_fixtures_requires_core_columns() -> None:
    fixtures = pd.DataFrame({"date": ["2026-06-23"]})

    with pytest.raises(ValueError, match="missing required columns"):
        filter_upcoming_fixtures(fixtures)
