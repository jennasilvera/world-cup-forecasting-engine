from __future__ import annotations

import pandas as pd

from wc_forecast.ledger.prediction_ledger import (
    LEDGER_COLUMNS,
    append_candidate_edges_to_prediction_ledger,
    append_prediction_ledger_row,
    build_prediction_ledger_row,
    build_prediction_ledger_row_from_batch_edge,
    match_outcome_from_score,
    realized_return_for_prediction,
    save_market_prediction_to_ledger,
    settle_prediction_ledger_row,
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


def test_match_outcome_from_score() -> None:
    assert match_outcome_from_score(2, 1) == "home_win"
    assert match_outcome_from_score(1, 1) == "draw"
    assert match_outcome_from_score(0, 1) == "away_win"


def test_realized_return_for_prediction() -> None:
    assert realized_return_for_prediction(
        best_outcome="draw",
        final_outcome="draw",
        home_odds=2.20,
        draw_odds=3.40,
        away_odds=3.50,
    ) == 2.40

    assert realized_return_for_prediction(
        best_outcome="draw",
        final_outcome="home_win",
        home_odds=2.20,
        draw_odds=3.40,
        away_odds=3.50,
    ) == -1.0


def test_settle_prediction_ledger_row_updates_result(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    ledger_path = tmp_path / "prediction_ledger.csv"

    _sample_results().to_csv(results_path, index=False)

    save_market_prediction_to_ledger(
        results_path=results_path,
        ledger_path=ledger_path,
        home_team="Argentina",
        away_team="France",
        home_odds=2.20,
        draw_odds=3.40,
        away_odds=3.50,
        prediction_timestamp="2026-06-19T12:00:00+00:00",
    )

    ledger = pd.read_csv(ledger_path)
    prediction_id = str(ledger.loc[0, "prediction_id"])

    destination = settle_prediction_ledger_row(
        ledger_path=ledger_path,
        prediction_id=prediction_id,
        final_home_score=1,
        final_away_score=1,
        closing_home_odds=2.10,
        closing_draw_odds=3.25,
        closing_away_odds=3.60,
    )

    settled = pd.read_csv(destination)

    assert settled.loc[0, "final_outcome"] == "draw"
    assert settled.loc[0, "realized_return"] == 2.40
    assert settled.loc[0, "closing_draw_odds"] == 3.25


def _sample_batch_edges() -> pd.DataFrame:
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
            "market_overround": [1.03, 1.04],
            "model_prob_home_win": [0.18, 0.17],
            "model_prob_draw": [0.53, 0.74],
            "model_prob_away_win": [0.29, 0.09],
            "market_fair_home_win": [0.44, 0.37],
            "market_fair_draw": [0.28, 0.29],
            "market_fair_away_win": [0.28, 0.34],
            "edge_home_win": [-0.26, -0.20],
            "edge_draw": [0.25, 0.45],
            "edge_away_win": [0.01, -0.25],
            "expected_value_home_win": [-0.60, -0.56],
            "expected_value_draw": [0.81, 1.43],
            "expected_value_away_win": [0.00, -0.73],
            "best_outcome": ["draw", "draw"],
            "best_edge": [0.25, 0.45],
            "best_expected_value": [0.81, 1.43],
            "decision": ["candidate_edge", "candidate_edge"],
            "ensemble_confidence": ["Medium", "High"],
            "ensemble_entropy": [0.91, 0.68],
            "max_model_disagreement": [0.78, 0.11],
        }
    )


def test_build_prediction_ledger_row_from_batch_edge() -> None:
    row = build_prediction_ledger_row_from_batch_edge(
        edge_row=_sample_batch_edges().iloc[0],
        prediction_timestamp="2026-06-19T12:00:00+00:00",
    )

    assert row["home_team"] == "Argentina"
    assert row["best_outcome"] == "draw"
    assert row["decision"] == "candidate_edge"
    assert set(LEDGER_COLUMNS).issubset(row)


def test_append_candidate_edges_to_prediction_ledger(tmp_path) -> None:
    batch_edges_path = tmp_path / "batch_market_edges.csv"
    ledger_path = tmp_path / "prediction_ledger.csv"

    _sample_batch_edges().to_csv(batch_edges_path, index=False)

    destination = append_candidate_edges_to_prediction_ledger(
        batch_edges_path=batch_edges_path,
        ledger_path=ledger_path,
        prediction_timestamp="2026-06-19T12:00:00+00:00",
    )

    saved = pd.read_csv(destination)

    assert destination.exists()
    assert len(saved) == 2
    assert set(saved["decision"]) == {"candidate_edge"}
