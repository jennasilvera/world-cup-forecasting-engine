from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.data_sources.international_results import (
    normalize_international_results,
    save_normalized_international_results,
    validate_international_results_source,
)


def _sample_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-06-01", "2024-06-05"],
            "home_team": ["Argentina", "France"],
            "away_team": ["Brazil", "Spain"],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "tournament": ["Friendly", "Friendly"],
            "city": ["Buenos Aires", "Paris"],
            "country": ["Argentina", "France"],
            "neutral": [False, "False"],
        }
    )


def test_validate_international_results_source_rejects_missing_columns() -> None:
    source = _sample_source().drop(columns=["neutral"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_international_results_source(source)


def test_normalize_international_results_outputs_engine_schema() -> None:
    normalized = normalize_international_results(_sample_source())

    assert list(normalized.columns) == [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
        "outcome",
    ]
    assert normalized.loc[0, "home_team"] == "Argentina"
    assert normalized.loc[0, "outcome"] == "home_win"
    assert normalized.loc[1, "outcome"] == "draw"
    assert bool(normalized.loc[1, "neutral"]) is False


def test_normalize_international_results_rejects_identical_teams() -> None:
    source = _sample_source()
    source.loc[0, "away_team"] = "Argentina"

    with pytest.raises(ValueError, match="identical teams"):
        normalize_international_results(source)


def test_save_normalized_international_results_writes_csv(tmp_path) -> None:
    source_path = tmp_path / "raw_results.csv"
    output_path = tmp_path / "normalized_results.csv"

    _sample_source().to_csv(source_path, index=False)

    destination = save_normalized_international_results(
        source_path=source_path,
        output_path=output_path,
    )

    saved = pd.read_csv(destination)

    assert destination.exists()
    assert len(saved) == 2
    assert "home_team" in saved.columns


def test_normalize_international_results_adds_away_win_outcome() -> None:
    source = _sample_source()
    source.loc[0, "home_score"] = 0
    source.loc[0, "away_score"] = 2

    normalized = normalize_international_results(source)

    assert normalized.loc[0, "outcome"] == "away_win"


def test_normalize_international_results_drops_incomplete_matches() -> None:
    source = pd.concat(
        [
            _sample_source(),
            pd.DataFrame(
                {
                    "date": ["2026-06-11"],
                    "home_team": ["Mexico"],
                    "away_team": ["South Africa"],
                    "home_score": [None],
                    "away_score": [None],
                    "tournament": ["FIFA World Cup"],
                    "city": ["Mexico City"],
                    "country": ["Mexico"],
                    "neutral": [False],
                }
            ),
        ],
        ignore_index=True,
    )

    normalized = normalize_international_results(source)

    assert len(normalized) == 2
    assert "Mexico" not in set(normalized["home_team"])
