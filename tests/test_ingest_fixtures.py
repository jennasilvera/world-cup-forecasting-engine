from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.data.ingest_fixtures import normalize_world_cup_fixtures


def test_normalize_world_cup_fixtures_adds_defaults() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-06-22"],
            "home_team": [" Argentina "],
            "away_team": [" Austria "],
        }
    )

    result = normalize_world_cup_fixtures(raw)

    assert result.loc[0, "date"].isoformat() == "2026-06-22"
    assert result.loc[0, "home_team"] == "Argentina"
    assert result.loc[0, "away_team"] == "Austria"
    assert result.loc[0, "tournament"] == "FIFA World Cup"
    assert bool(result.loc[0, "neutral"])
    assert result.loc[0, "status"] == "Scheduled"


def test_normalize_world_cup_fixtures_parses_optional_columns() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-06-22"],
            "home_team": ["France"],
            "away_team": ["Iraq"],
            "tournament": ["FIFA World Cup"],
            "neutral": ["yes"],
            "status": ["Scheduled"],
        }
    )

    result = normalize_world_cup_fixtures(raw)

    assert bool(result.loc[0, "neutral"])


def test_normalize_world_cup_fixtures_rejects_missing_required_columns() -> None:
    raw = pd.DataFrame({"date": ["2026-06-22"]})

    with pytest.raises(ValueError, match="missing required columns"):
        normalize_world_cup_fixtures(raw)


def test_normalize_world_cup_fixtures_rejects_blank_teams() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-06-22"],
            "home_team": [""],
            "away_team": ["Iraq"],
        }
    )

    with pytest.raises(ValueError, match="blank home_team"):
        normalize_world_cup_fixtures(raw)
