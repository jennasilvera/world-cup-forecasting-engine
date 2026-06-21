from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wc_forecast.reports.forecast_audit import (
    build_forecast_audit,
    save_forecast_audit,
)


def _sample_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-06-21", "2026-06-22"],
            "home_team": ["Belgium", "Argentina"],
            "away_team": ["Iran", "Austria"],
            "predicted_outcome": ["home_win", "away_win"],
            "predicted_winner": ["Belgium", "Austria"],
            "model_confidence": [0.46, 0.62],
            "home_rating_source": ["rating_lookup", "rating_lookup"],
            "away_rating_source": ["rating_lookup", "alias_lookup"],
            "rating_warning": ["", "alias used"],
        }
    )


def test_build_forecast_audit_summarizes_forecasts() -> None:
    audit = build_forecast_audit(_sample_forecasts())

    metrics = dict(zip(audit["metric"], audit["value"], strict=True))

    assert metrics["forecast_count"] == 2
    assert metrics["average_confidence"] == 0.54
    assert metrics["highest_confidence"] == 0.62
    assert metrics["lowest_confidence"] == 0.46
    assert metrics["rating_warning_count"] == 1
    assert metrics["alias_lookup_count"] == 1
    assert metrics["predicted_home_win_count"] == 1
    assert metrics["predicted_away_win_count"] == 1


def test_build_forecast_audit_rejects_missing_columns() -> None:
    forecasts = pd.DataFrame({"date": ["2026-06-21"]})

    with pytest.raises(ValueError, match="missing required columns"):
        build_forecast_audit(forecasts)


def test_save_forecast_audit_writes_csv(tmp_path: Path) -> None:
    forecasts_path = tmp_path / "forecasts.csv"
    output_path = tmp_path / "audit.csv"

    _sample_forecasts().to_csv(forecasts_path, index=False)

    audit = save_forecast_audit(
        forecasts_path=forecasts_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(audit) > 0
