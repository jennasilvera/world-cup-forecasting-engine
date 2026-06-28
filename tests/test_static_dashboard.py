from __future__ import annotations

import pandas as pd

from wc_forecast.dashboard.static_dashboard import (
    build_dashboard_from_prediction_ledger,
    build_dashboard_html,
)
from wc_forecast.storage.database import initialize_database
from wc_forecast.storage.prediction_store import write_forecasts_to_prediction_ledger


def test_build_dashboard_html_contains_match_and_portfolio_sections() -> None:
    rows = [
        {
            "match_date": "2026-06-20",
            "home_team": "Argentina",
            "away_team": "Brazil",
            "predicted_outcome": "home_win",
            "predicted_winner": "Argentina",
            "model_confidence": 0.61,
            "prob_home_win": 0.61,
            "prob_draw": 0.21,
            "prob_away_win": 0.18,
            "status": "open",
        }
    ]

    html = build_dashboard_html(rows)

    assert "Portfolio view" in html
    assert "Match view" in html
    assert "Argentina vs Brazil" in html


def test_build_dashboard_from_prediction_ledger_writes_html(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    output_path = tmp_path / "dashboard.html"

    initialize_database(database_path)

    forecasts = pd.DataFrame(
        {
            "date": ["2026-06-20"],
            "kickoff_at": ["2026-06-20T18:00:00Z"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "predicted_outcome": ["home_win"],
            "predicted_winner": ["Argentina"],
            "model_confidence": [0.61],
            "prob_home_win": [0.61],
            "prob_draw": [0.21],
            "prob_away_win": [0.18],
        }
    )

    write_forecasts_to_prediction_ledger(forecasts, database_path=database_path)

    rows = build_dashboard_from_prediction_ledger(
        output_path=output_path,
        database_path=database_path,
    )

    assert len(rows) == 1
    assert output_path.exists()
    assert "World Cup Forecasting Dashboard" in output_path.read_text()
