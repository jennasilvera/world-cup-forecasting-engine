from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.data.ingest_results import load_historical_results, validate_results_frame


def test_load_historical_results_adds_outcome_and_sorts(tmp_path) -> None:
    raw_file = tmp_path / "results.csv"
    raw_file.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,True\n"
        "2022-12-17,Croatia,Morocco,2,1,FIFA World Cup,Al Rayyan,Qatar,True\n"
    )

    results = load_historical_results(raw_file)

    assert list(results["home_team"]) == ["Croatia", "Argentina"]
    assert list(results["outcome"]) == ["home_win", "draw"]
    assert results["home_score"].dtype == "int64"
    assert results["away_score"].dtype == "int64"


def test_validate_results_frame_rejects_missing_columns() -> None:
    bad_results = pd.DataFrame(
        {
            "date": ["2022-12-18"],
            "home_team": ["Argentina"],
            "away_team": ["France"],
        }
    )

    with pytest.raises(ValueError, match="Missing required result columns"):
        validate_results_frame(bad_results)


def test_validate_results_frame_rejects_team_playing_itself() -> None:
    bad_results = pd.DataFrame(
        {
            "date": ["2022-12-18"],
            "home_team": ["Argentina"],
            "away_team": ["Argentina"],
            "home_score": [1],
            "away_score": [1],
            "tournament": ["FIFA World Cup"],
            "city": ["Lusail"],
            "country": ["Qatar"],
            "neutral": [True],
        }
    )

    with pytest.raises(ValueError, match="A team cannot play itself"):
        validate_results_frame(bad_results)


def test_validate_results_frame_rejects_negative_scores() -> None:
    bad_results = pd.DataFrame(
        {
            "date": ["2022-12-18"],
            "home_team": ["Argentina"],
            "away_team": ["France"],
            "home_score": [-1],
            "away_score": [1],
            "tournament": ["FIFA World Cup"],
            "city": ["Lusail"],
            "country": ["Qatar"],
            "neutral": [True],
        }
    )

    with pytest.raises(ValueError, match="Scores cannot be negative"):
        validate_results_frame(bad_results)
