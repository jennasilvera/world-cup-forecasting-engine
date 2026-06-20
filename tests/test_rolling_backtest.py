from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.features.build_features import build_match_features
from wc_forecast.validation.rolling_backtest import (
    run_rolling_backtest,
    summarize_rolling_backtest,
)


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
                    "2021-01-01",
                    "2021-02-01",
                ]
            ),
            "home_team": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
            ],
            "away_team": [
                "B",
                "C",
                "A",
                "C",
                "A",
                "B",
                "B",
                "C",
                "A",
                "C",
                "A",
                "B",
            ],
            "home_score": [2, 1, 1, 0, 3, 1, 2, 1, 0, 2, 1, 1],
            "away_score": [0, 1, 2, 1, 0, 1, 1, 2, 0, 0, 1, 2],
            "tournament": ["Friendly"] * 12,
            "city": ["X"] * 12,
            "country": ["Y"] * 12,
            "neutral": [True] * 12,
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
                "draw",
                "away_win",
            ],
        }
    )


def test_run_rolling_backtest_returns_model_fold_metrics() -> None:
    features = build_match_features(_sample_results())

    result = run_rolling_backtest(
        features=features,
        cutoff_dates=["2020-01-01"],
        model_types=["logistic"],
        evaluation_window_days=365,
    )

    assert len(result) == 1
    assert result.loc[0, "model_type"] == "logistic"
    assert result.loc[0, "train_rows"] == 6
    assert result.loc[0, "test_rows"] == 4
    assert result.loc[0, "log_loss"] > 0


def test_run_rolling_backtest_supports_multiple_models() -> None:
    features = build_match_features(_sample_results())

    result = run_rolling_backtest(
        features=features,
        cutoff_dates=["2020-01-01"],
        model_types=["logistic", "gradient_boosting"],
        evaluation_window_days=365,
    )

    assert set(result["model_type"]) == {"logistic", "gradient_boosting"}


def test_summarize_rolling_backtest_orders_by_log_loss() -> None:
    metrics = pd.DataFrame(
        {
            "model_type": ["a", "b"],
            "cutoff_date": ["2020-01-01", "2020-01-01"],
            "evaluation_window_days": [365, 365],
            "train_rows": [10, 10],
            "test_rows": [5, 5],
            "accuracy": [0.5, 0.6],
            "log_loss": [0.9, 0.8],
            "multiclass_brier_score": [0.5, 0.4],
        }
    )

    summary = summarize_rolling_backtest(metrics)

    assert summary.loc[0, "model_type"] == "b"


def test_run_rolling_backtest_rejects_empty_valid_folds() -> None:
    features = build_match_features(_sample_results())

    with pytest.raises(ValueError, match="no valid folds"):
        run_rolling_backtest(
            features=features,
            cutoff_dates=["1900-01-01"],
            model_types=["logistic"],
            evaluation_window_days=365,
        )
