from __future__ import annotations

import pandas as pd

from wc_forecast.ledger.prediction_ledger import (
    LEDGER_COLUMNS,
    append_prediction_ledger_row,
    build_prediction_ledger_row,
    save_market_prediction_to_ledger,
)
from wc_forecast.models.market import calculate_market_edge, calculate_market_probabilities


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


def _sample_prediction() -> dict[str, object]:
    return {
        "home_team": "Argentina",
        "away_team": "France",
        "tournament": "FIFA World Cup",
        "ensemble_prob_home_win": 0.30,
        "ensemble_prob_draw": 0.25,
        "ensemble_prob_away_win": 0.45,
        "ensemble_confidence": "Medium",
        "ensemble_entropy": 0.90,
        "max_model_disagreement": 0.15,
    }


def test_build_prediction_ledger_row_has_expected_columns() -> None:
    market = calculate_market_probabilities(
        home_win_odds=2.20,
        draw_odds=3.40,
        away_win_odds=3.50,
    )
    edge = calculate_market_edge(
        model_prob_home_win=0.30,
        model_prob_draw=0.25,
        model_prob_away_win=0.45,
        home_win_odds=2.20,
        draw_odds=3.40,
        away_win_odds=3.50,
    )

    row = build_prediction_ledger_row(
        prediction=_sample_prediction(),
        market=market,
        edge=edge,
        home_odds=2.20,
        draw_odds=3.40,
        away_odds=3.50,
        prediction_timestamp="2026-06-19T12:00:00+00:00",
    )

    assert set(LEDGER_COLUMNS).issubset(row)
    assert row["home_team"] == "Argentina"
    assert row["decision"] in {"candidate_edge", "no_edge"}


def test_append_prediction_ledger_row_writes_csv(tmp_path) -> None:
    ledger_path = tmp_path / "prediction_ledger.csv"
    market = calculate_market_probabilities(2.20, 3.40, 3.50)
    edge = calculate_market_edge(0.30, 0.25, 0.45, 2.20, 3.40, 3.50)
    row = build_prediction_ledger_row(
        prediction=_sample_prediction(),
        market=market,
        edge=edge,
        home_odds=2.20,
        draw_odds=3.40,
        away_odds=3.50,
    )

    destination = append_prediction_ledger_row(
        ledger_path=ledger_path,
        row=row,
    )

    saved = pd.read_csv(destination)

    assert destination.exists()
    assert len(saved) == 1
    assert "prediction_id" in saved.columns


def test_save_market_prediction_to_ledger_appends_forecast(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    ledger_path = tmp_path / "prediction_ledger.csv"

    _sample_results().to_csv(results_path, index=False)

    destination = save_market_prediction_to_ledger(
        results_path=results_path,
        ledger_path=ledger_path,
        home_team="Argentina",
        away_team="France",
        home_odds=2.20,
        draw_odds=3.40,
        away_odds=3.50,
        prediction_timestamp="2026-06-19T12:00:00+00:00",
    )

    saved = pd.read_csv(destination)

    assert destination.exists()
    assert len(saved) == 1
    assert saved.loc[0, "home_team"] == "Argentina"
    assert saved.loc[0, "away_team"] == "France"
