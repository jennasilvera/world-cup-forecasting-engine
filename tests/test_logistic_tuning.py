from __future__ import annotations

import pandas as pd

from wc_forecast.features.build_features import build_match_features
from wc_forecast.validation.logistic_tuning import tune_logistic_model


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2019-01-01",
                    "2019-02-01",
                    "2019-03-01",
                    "2019-04-01",
                    "2019-05-01",
                    "2019-06-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                ]
            ),
            "home_team": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "away_team": ["B", "C", "A", "C", "A", "B", "B", "C", "A", "C"],
            "home_score": [2, 1, 1, 0, 3, 1, 2, 1, 0, 2],
            "away_score": [0, 1, 2, 1, 0, 1, 1, 2, 0, 0],
            "tournament": ["Friendly"] * 10,
            "city": ["X"] * 10,
            "country": ["Y"] * 10,
            "neutral": [True] * 10,
            "outcome": [
                "home_win",
                "draw",
                "away_win",
                "away_win",
                "home_win",
                "draw",
                "home_win",
                "away_win",
                "draw",
                "home_win",
            ],
        }
    )


def test_tune_logistic_model_returns_sorted_grid_results() -> None:
    features = build_match_features(_sample_results())

    result = tune_logistic_model(
        features=features,
        half_life_days_grid=[365.0, 730.0],
        logistic_c_grid=[0.5, 1.0],
        cutoff_dates=["2020-01-01"],
        evaluation_window_days=365,
    )

    assert len(result) == 4
    assert set(result["sample_weight_half_life_days"]) == {365.0, 730.0}
    assert set(result["logistic_c"]) == {0.5, 1.0}
    assert result["mean_log_loss"].is_monotonic_increasing
