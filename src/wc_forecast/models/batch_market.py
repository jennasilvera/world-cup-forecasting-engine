from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.models.market import (
    calculate_market_edge,
    calculate_market_probabilities,
)
from wc_forecast.reporting.match_report import generate_match_prediction

REQUIRED_MARKET_ODDS_COLUMNS = [
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
]



def _outcome_from_scores(home_score: int, away_score: int) -> str:
    """Return home_win, draw, or away_win from scores."""

    if home_score > away_score:
        return "home_win"

    if home_score < away_score:
        return "away_win"

    return "draw"


def _ensure_outcome_column(results: pd.DataFrame) -> pd.DataFrame:
    """Ensure results include an outcome column for downstream feature builders."""

    if "outcome" in results.columns:
        return results

    enriched = results.copy()
    enriched["outcome"] = enriched.apply(
        lambda row: _outcome_from_scores(
            home_score=int(row["home_score"]),
            away_score=int(row["away_score"]),
        ),
        axis=1,
    )

    return enriched

def _has_value(value: object) -> bool:
    """Return whether a CSV value is meaningfully populated."""

    return str(value).strip() != "" and str(value).strip().lower() != "nan"


def _optional_string(
    row: pd.Series,
    column: str,
    default: str,
) -> str:
    """Return a clean optional string from a row."""

    value = row.get(column, default)

    if not _has_value(value):
        return default

    return str(value).strip()


def validate_market_odds_slate(odds: pd.DataFrame) -> None:
    """Validate market odds input for batch evaluation."""

    missing_columns = sorted(set(REQUIRED_MARKET_ODDS_COLUMNS) - set(odds.columns))

    if missing_columns:
        raise ValueError(f"Market odds slate missing required columns: {missing_columns}")

    if odds.empty:
        raise ValueError("Market odds slate is empty.")

    for row_number, row in odds.iterrows():
        home_team = str(row["home_team"]).strip()
        away_team = str(row["away_team"]).strip()

        if not home_team:
            raise ValueError(f"Row {row_number} has blank home_team.")

        if not away_team:
            raise ValueError(f"Row {row_number} has blank away_team.")

        if home_team == away_team:
            raise ValueError(f"Row {row_number} has identical teams: {home_team}.")

        for column in ["home_odds", "draw_odds", "away_odds"]:
            value = float(row[column])

            if value <= 1.0:
                raise ValueError(
                    f"Row {row_number} has invalid decimal odds in {column}: {value}"
                )


def evaluate_market_odds_slate(
    results: pd.DataFrame,
    odds: pd.DataFrame,
    default_tournament: str = "FIFA World Cup",
    minimum_edge: float = 0.03,
    minimum_expected_value: float = 0.02,
) -> pd.DataFrame:
    """Evaluate a slate of market odds against model probabilities."""

    validate_market_odds_slate(odds)
    results = _ensure_outcome_column(results)

    rows: list[dict[str, object]] = []

    for _, market_row in odds.iterrows():
        home_team = str(market_row["home_team"]).strip()
        away_team = str(market_row["away_team"]).strip()
        tournament = _optional_string(
            row=market_row,
            column="tournament",
            default=default_tournament,
        )
        kickoff_timestamp = _optional_string(
            row=market_row,
            column="kickoff_timestamp",
            default="",
        )

        home_odds = float(market_row["home_odds"])
        draw_odds = float(market_row["draw_odds"])
        away_odds = float(market_row["away_odds"])

        prediction = generate_match_prediction(
            results=results,
            home_team=home_team,
            away_team=away_team,
            tournament=tournament,
        )

        market = calculate_market_probabilities(
            home_win_odds=home_odds,
            draw_odds=draw_odds,
            away_win_odds=away_odds,
        )
        edge = calculate_market_edge(
            model_prob_home_win=float(prediction["ensemble_prob_home_win"]),
            model_prob_draw=float(prediction["ensemble_prob_draw"]),
            model_prob_away_win=float(prediction["ensemble_prob_away_win"]),
            home_win_odds=home_odds,
            draw_odds=draw_odds,
            away_win_odds=away_odds,
            minimum_edge=minimum_edge,
            minimum_expected_value=minimum_expected_value,
        )

        rows.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "tournament": tournament,
                "kickoff_timestamp": kickoff_timestamp,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "market_overround": market.overround,
                "model_prob_home_win": prediction["ensemble_prob_home_win"],
                "model_prob_draw": prediction["ensemble_prob_draw"],
                "model_prob_away_win": prediction["ensemble_prob_away_win"],
                "market_fair_home_win": market.fair_home_win,
                "market_fair_draw": market.fair_draw,
                "market_fair_away_win": market.fair_away_win,
                "edge_home_win": edge.edge_home_win,
                "edge_draw": edge.edge_draw,
                "edge_away_win": edge.edge_away_win,
                "expected_value_home_win": edge.expected_value_home_win,
                "expected_value_draw": edge.expected_value_draw,
                "expected_value_away_win": edge.expected_value_away_win,
                "best_outcome": edge.best_outcome,
                "best_edge": edge.best_edge,
                "best_expected_value": edge.best_expected_value,
                "decision": edge.decision,
                "ensemble_confidence": prediction["ensemble_confidence"],
                "ensemble_entropy": prediction["ensemble_entropy"],
                "max_model_disagreement": prediction["max_model_disagreement"],
            }
        )

    evaluation = pd.DataFrame(rows)

    decision_rank = {"candidate_edge": 0, "no_edge": 1}
    evaluation["_decision_rank"] = evaluation["decision"].map(decision_rank).fillna(9)
    evaluation = evaluation.sort_values(
        by=["_decision_rank", "best_expected_value", "best_edge"],
        ascending=[True, False, False],
    )
    evaluation = evaluation.drop(columns=["_decision_rank"]).reset_index(drop=True)

    return evaluation


def save_market_odds_slate_evaluation(
    results: pd.DataFrame,
    odds: pd.DataFrame,
    output_path: str | Path,
    default_tournament: str = "FIFA World Cup",
    minimum_edge: float = 0.03,
    minimum_expected_value: float = 0.02,
) -> Path:
    """Evaluate a slate and save results as CSV."""

    evaluation = evaluate_market_odds_slate(
        results=results,
        odds=odds,
        default_tournament=default_tournament,
        minimum_edge=minimum_edge,
        minimum_expected_value=minimum_expected_value,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(destination, index=False)

    return destination
