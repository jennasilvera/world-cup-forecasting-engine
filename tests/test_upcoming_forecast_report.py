from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from wc_forecast.reports.upcoming_forecast_report import (
    build_upcoming_forecast_report,
    save_upcoming_forecast_report,
)


def _sample_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-06-21", "2026-06-22"],
            "home_team": ["Belgium", "Argentina"],
            "away_team": ["Iran", "Austria"],
            "home_elo_rating": [1873.0, 1950.0],
            "away_elo_rating": [1822.0, 1900.0],
            "prob_home_win": [0.46, 0.60],
            "prob_draw": [0.30, 0.25],
            "prob_away_win": [0.24, 0.15],
            "predicted_winner": ["Belgium", "Argentina"],
            "model_confidence": [0.46, 0.60],
            "rating_warning": ["", ""],
        }
    )


def test_build_upcoming_forecast_report_contains_sections() -> None:
    report = build_upcoming_forecast_report(_sample_forecasts())

    assert "# Upcoming World Cup Forecast Report" in report
    assert "Highest-Confidence Forecasts" in report
    assert "Most Uncertain Matches" in report
    assert "Potential Upset Watch" in report
    assert "Belgium vs Iran" in report


def test_build_upcoming_forecast_report_rejects_missing_columns() -> None:
    forecasts = pd.DataFrame({"date": ["2026-06-21"]})

    with pytest.raises(ValueError, match="missing required columns"):
        build_upcoming_forecast_report(forecasts)


def test_save_upcoming_forecast_report_writes_file(tmp_path: Path) -> None:
    forecasts_path = tmp_path / "forecasts.csv"
    output_path = tmp_path / "report.md"

    _sample_forecasts().to_csv(forecasts_path, index=False)

    report = save_upcoming_forecast_report(
        forecasts_path=forecasts_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.read_text() == report
    assert "Matches forecasted" in report
