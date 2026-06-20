from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.models.batch_market import (
    evaluate_market_odds_slate,
    save_market_odds_slate_evaluation,
    validate_market_odds_slate,
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
        }
    )


def _sample_odds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["Argentina", "Brazil"],
            "away_team": ["France", "Spain"],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "kickoff_timestamp": [
                "2026-06-20T19:00:00+00:00",
                "2026-06-21T19:00:00+00:00",
            ],
            "home_odds": [2.20, 2.60],
            "draw_odds": [3.40, 3.30],
            "away_odds": [3.50, 2.80],
        }
    )


def test_validate_market_odds_slate_rejects_missing_columns() -> None:
    odds = _sample_odds().drop(columns=["home_odds"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_market_odds_slate(odds)


def test_validate_market_odds_slate_rejects_identical_teams() -> None:
    odds = _sample_odds()
    odds.loc[0, "away_team"] = "Argentina"

    with pytest.raises(ValueError, match="identical teams"):
        validate_market_odds_slate(odds)


def test_evaluate_market_odds_slate_returns_ranked_edges() -> None:
    evaluation = evaluate_market_odds_slate(
        results=_sample_results(),
        odds=_sample_odds(),
    )

    assert len(evaluation) == 2
    assert "best_expected_value" in evaluation.columns
    assert "decision" in evaluation.columns
    assert set(evaluation["home_team"]) == {"Argentina", "Brazil"}


def test_save_market_odds_slate_evaluation_writes_csv(tmp_path) -> None:
    output_path = tmp_path / "batch_market_edges.csv"

    destination = save_market_odds_slate_evaluation(
        results=_sample_results(),
        odds=_sample_odds(),
        output_path=output_path,
    )

    saved = pd.read_csv(destination)

    assert destination.exists()
    assert len(saved) == 2
    assert "best_outcome" in saved.columns
