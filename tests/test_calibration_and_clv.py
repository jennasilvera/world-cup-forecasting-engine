from __future__ import annotations

import pandas as pd

from wc_forecast.reports.calibration import build_multiclass_calibration_table
from wc_forecast.reports.closing_line_value import (
    build_clv_table,
    calculate_selected_outcome_clv,
)


def test_build_multiclass_calibration_table() -> None:
    predictions = pd.DataFrame(
        {
            "prob_home_win": [0.7, 0.2, 0.4],
            "prob_draw": [0.2, 0.3, 0.4],
            "prob_away_win": [0.1, 0.5, 0.2],
            "realized_outcome": ["home_win", "away_win", "draw"],
        }
    )

    calibration = build_multiclass_calibration_table(predictions, n_bins=5)

    assert set(calibration["outcome"]) == {"home_win", "draw", "away_win"}
    assert {
        "forecast_count",
        "average_forecast_probability",
        "observed_frequency",
        "calibration_error",
    } <= set(calibration.columns)


def test_calculate_selected_outcome_clv_positive_when_price_shortens() -> None:
    row = pd.Series(
        {
            "predicted_outcome": "home_win",
            "market_home_odds": 2.20,
            "closing_home_odds": 2.00,
        }
    )

    clv = calculate_selected_outcome_clv(row)

    assert clv is not None
    assert round(clv, 4) == 0.1000


def test_build_clv_table() -> None:
    predictions = pd.DataFrame(
        {
            "prediction_id": ["pred_1"],
            "match_date": ["2026-06-20"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "predicted_outcome": ["home_win"],
            "predicted_winner": ["Argentina"],
            "model_confidence": [0.61],
            "market_home_odds": [2.20],
            "market_draw_odds": [3.40],
            "market_away_odds": [3.50],
            "closing_home_odds": [2.00],
            "closing_draw_odds": [3.60],
            "closing_away_odds": [4.00],
        }
    )

    clv = build_clv_table(predictions)

    assert clv.loc[0, "closing_line_value"] > 0
    assert clv.loc[0, "market_implied_probability"] < clv.loc[0, "closing_implied_probability"]
