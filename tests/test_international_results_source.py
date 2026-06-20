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
    ]
    assert normalized.loc[0, "home_team"] == "Argentina"
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
