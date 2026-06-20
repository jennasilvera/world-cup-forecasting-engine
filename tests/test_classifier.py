from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.features.build_features import build_match_features
from wc_forecast.models.classifier import (
    OUTCOME_ORDER,
    PREDICTION_COLUMNS,
    chronological_train_test_split,
    run_logistic_backtest,
    train_logistic_regression,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-11-20",
                    "2022-11-21",
                    "2022-11-21",
                    "2022-11-22",
                    "2022-11-22",
                    "2022-11-23",
                    "2022-11-24",
                    "2022-11-25",
                    "2022-12-03",
                    "2022-12-18",
                ]
            ),
            "home_team": [
                "Qatar",
                "England",
                "United States",
                "Argentina",
                "Mexico",
                "Spain",
                "Brazil",
                "Netherlands",
                "Netherlands",
                "Argentina",
            ],
            "away_team": [
                "Ecuador",
                "Iran",
                "Wales",
                "Saudi Arabia",
                "Poland",
                "Costa Rica",
                "Serbia",
                "Ecuador",
                "United States",
                "France",
            ],
            "home_score": [0, 6, 1, 1, 0, 7, 2, 1, 3, 3],
            "away_score": [2, 2, 1, 2, 0, 0, 0, 1, 1, 3],
            "tournament": ["FIFA World Cup"] * 10,
            "city": [
                "Al Khor",
                "Al Rayyan",
                "Al Rayyan",
                "Lusail",
                "Doha",
                "Doha",
                "Lusail",
                "Doha",
                "Khalifa",
                "Lusail",
            ],
            "country": ["Qatar"] * 10,
            "neutral": [False, True, True, True, True, True, True, True, True, True],
            "outcome": [
                "away_win",
                "home_win",
                "draw",
                "away_win",
                "draw",
                "home_win",
                "home_win",
                "draw",
                "home_win",
                "draw",
            ],
        }
    )


def _sample_features() -> pd.DataFrame:
    return build_match_features(_sample_results())


def test_chronological_train_test_split_preserves_order() -> None:
    train, test = chronological_train_test_split(_sample_features(), test_fraction=0.30)

    assert len(train) == 7
    assert len(test) == 3
    assert train["date"].max() <= test["date"].min()


def test_train_logistic_regression_requires_all_classes() -> None:
    features = _sample_features()
    features_without_draws = features[features["outcome"] != "draw"].copy()

    with pytest.raises(ValueError, match="missing required outcome classes"):
        train_logistic_regression(features_without_draws)


def test_run_logistic_backtest_outputs_probabilities_and_metrics() -> None:
    result = run_logistic_backtest(_sample_features(), test_fraction=0.30)

    assert len(result.predictions) == 3
    assert set(PREDICTION_COLUMNS).issubset(result.predictions.columns)

    probability_sums = result.predictions[PREDICTION_COLUMNS].sum(axis=1)
    assert probability_sums.round(8).eq(1.0).all()

    metric_names = set(result.metrics["metric"])
    assert {"accuracy", "log_loss", "multiclass_brier_score"}.issubset(metric_names)


def test_outcome_order_is_three_class_football_result() -> None:
    assert OUTCOME_ORDER == ["home_win", "draw", "away_win"]
