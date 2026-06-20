from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from wc_forecast.data.ingest_results import load_historical_results
from wc_forecast.models.market import (
    MarketEdge,
    MarketProbabilities,
    calculate_market_edge,
    calculate_market_probabilities,
)
from wc_forecast.reporting.match_report import generate_match_prediction

LEDGER_COLUMNS = [
    "prediction_id",
    "model_version",
    "feature_version",
    "prediction_timestamp",
    "kickoff_timestamp",
    "home_team",
    "away_team",
    "tournament",
    "model_prob_home_win",
    "model_prob_draw",
    "model_prob_away_win",
    "market_fair_home_win",
    "market_fair_draw",
    "market_fair_away_win",
    "home_odds",
    "draw_odds",
    "away_odds",
    "market_overround",
    "edge_home_win",
    "edge_draw",
    "edge_away_win",
    "expected_value_home_win",
    "expected_value_draw",
    "expected_value_away_win",
    "best_outcome",
    "best_edge",
    "best_expected_value",
    "decision",
    "ensemble_confidence",
    "ensemble_entropy",
    "max_model_disagreement",
    "full_kelly_fraction",
    "suggested_stake_fraction",
    "suggested_stake_amount",
    "stake_sizing_reason",
    "closing_home_odds",
    "closing_draw_odds",
    "closing_away_odds",
    "final_home_score",
    "final_away_score",
    "final_outcome",
    "realized_return",
    "notes",
]


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(UTC).isoformat()


def build_prediction_ledger_row(
    prediction: dict[str, object],
    market: MarketProbabilities,
    edge: MarketEdge,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    model_version: str = "demo-v1",
    feature_version: str = "demo-features-v1",
    prediction_timestamp: str | None = None,
    kickoff_timestamp: str = "",
    notes: str = "",
) -> dict[str, object]:
    """Build one prediction ledger row."""

    return {
        "prediction_id": str(uuid4()),
        "model_version": model_version,
        "feature_version": feature_version,
        "prediction_timestamp": prediction_timestamp or utc_now_iso(),
        "kickoff_timestamp": kickoff_timestamp,
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "tournament": prediction["tournament"],
        "model_prob_home_win": prediction["ensemble_prob_home_win"],
        "model_prob_draw": prediction["ensemble_prob_draw"],
        "model_prob_away_win": prediction["ensemble_prob_away_win"],
        "market_fair_home_win": market.fair_home_win,
        "market_fair_draw": market.fair_draw,
        "market_fair_away_win": market.fair_away_win,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "market_overround": market.overround,
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
        "full_kelly_fraction": "",
        "suggested_stake_fraction": "",
        "suggested_stake_amount": "",
        "stake_sizing_reason": "",
        "closing_home_odds": "",
        "closing_draw_odds": "",
        "closing_away_odds": "",
        "final_home_score": "",
        "final_away_score": "",
        "final_outcome": "",
        "realized_return": "",
        "notes": notes,
    }


def append_prediction_ledger_row(
    ledger_path: str | Path,
    row: dict[str, object],
) -> Path:
    """Append one row to the prediction ledger CSV."""

    destination = Path(ledger_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([row], columns=LEDGER_COLUMNS)

    if destination.exists() and destination.stat().st_size > 0:
        existing = pd.read_csv(destination)
        ledger = pd.concat([existing, new_row], ignore_index=True)
    else:
        ledger = new_row

    ledger.to_csv(destination, index=False)

    return destination


def save_market_prediction_to_ledger(
    results_path: str | Path,
    ledger_path: str | Path,
    home_team: str,
    away_team: str,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    tournament: str = "FIFA World Cup",
    model_version: str = "demo-v1",
    feature_version: str = "demo-features-v1",
    prediction_timestamp: str | None = None,
    kickoff_timestamp: str = "",
    minimum_edge: float = 0.03,
    minimum_expected_value: float = 0.02,
    notes: str = "",
) -> Path:
    """Generate a market-aware forecast and append it to the ledger."""

    results = load_historical_results(results_path)
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

    row = build_prediction_ledger_row(
        prediction=prediction,
        market=market,
        edge=edge,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        model_version=model_version,
        feature_version=feature_version,
        prediction_timestamp=prediction_timestamp,
        kickoff_timestamp=kickoff_timestamp,
        notes=notes,
    )

    return append_prediction_ledger_row(
        ledger_path=ledger_path,
        row=row,
    )


def match_outcome_from_score(home_score: int, away_score: int) -> str:
    """Return home_win, draw, or away_win from a final score."""

    if home_score > away_score:
        return "home_win"

    if home_score < away_score:
        return "away_win"

    return "draw"


def realized_return_for_prediction(
    best_outcome: str,
    final_outcome: str,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    stake: float = 1.0,
) -> float:
    """Calculate flat-stake realized return for the logged best outcome."""

    odds_by_outcome = {
        "home_win": home_odds,
        "draw": draw_odds,
        "away_win": away_odds,
    }

    if best_outcome not in odds_by_outcome:
        raise ValueError(f"Invalid best_outcome: {best_outcome}")

    if final_outcome not in odds_by_outcome:
        raise ValueError(f"Invalid final_outcome: {final_outcome}")

    if stake <= 0.0:
        raise ValueError("stake must be positive.")

    if best_outcome == final_outcome:
        return stake * (odds_by_outcome[best_outcome] - 1.0)

    return -stake


def settle_prediction_ledger_row(
    ledger_path: str | Path,
    prediction_id: str,
    final_home_score: int,
    final_away_score: int,
    closing_home_odds: float | None = None,
    closing_draw_odds: float | None = None,
    closing_away_odds: float | None = None,
    stake: float = 1.0,
) -> Path:
    """Update one prediction ledger row with final result and realized return."""

    path = Path(ledger_path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction ledger not found: {path}")

    ledger = pd.read_csv(path, keep_default_na=False, dtype=object)

    for column in LEDGER_COLUMNS:
        if column not in ledger.columns:
            ledger[column] = ""

    for column in LEDGER_COLUMNS:
        ledger[column] = ledger[column].astype("object")

    if "prediction_id" not in ledger.columns:
        raise ValueError("Prediction ledger missing prediction_id column.")

    matches = ledger["prediction_id"] == prediction_id

    if not matches.any():
        raise ValueError(f"Prediction id not found: {prediction_id}")

    final_outcome = match_outcome_from_score(
        home_score=final_home_score,
        away_score=final_away_score,
    )

    row_index = ledger.index[matches][0]
    best_outcome = str(ledger.loc[row_index, "best_outcome"])

    realized_return = realized_return_for_prediction(
        best_outcome=best_outcome,
        final_outcome=final_outcome,
        home_odds=float(ledger.loc[row_index, "home_odds"]),
        draw_odds=float(ledger.loc[row_index, "draw_odds"]),
        away_odds=float(ledger.loc[row_index, "away_odds"]),
        stake=stake,
    )

    ledger.loc[row_index, "final_home_score"] = final_home_score
    ledger.loc[row_index, "final_away_score"] = final_away_score
    ledger.loc[row_index, "final_outcome"] = final_outcome
    ledger.loc[row_index, "realized_return"] = realized_return

    if closing_home_odds is not None:
        ledger.loc[row_index, "closing_home_odds"] = closing_home_odds

    if closing_draw_odds is not None:
        ledger.loc[row_index, "closing_draw_odds"] = closing_draw_odds

    if closing_away_odds is not None:
        ledger.loc[row_index, "closing_away_odds"] = closing_away_odds

    ledger.to_csv(path, index=False)

    return path


def build_prediction_ledger_row_from_batch_edge(
    edge_row: pd.Series,
    model_version: str = "demo-v1",
    feature_version: str = "demo-features-v1",
    prediction_timestamp: str | None = None,
    notes: str = "",
) -> dict[str, object]:
    """Build one prediction ledger row from a batch market edge row."""

    return {
        "prediction_id": str(uuid4()),
        "model_version": model_version,
        "feature_version": feature_version,
        "prediction_timestamp": prediction_timestamp or utc_now_iso(),
        "kickoff_timestamp": edge_row.get("kickoff_timestamp", ""),
        "home_team": edge_row["home_team"],
        "away_team": edge_row["away_team"],
        "tournament": edge_row["tournament"],
        "model_prob_home_win": edge_row["model_prob_home_win"],
        "model_prob_draw": edge_row["model_prob_draw"],
        "model_prob_away_win": edge_row["model_prob_away_win"],
        "market_fair_home_win": edge_row["market_fair_home_win"],
        "market_fair_draw": edge_row["market_fair_draw"],
        "market_fair_away_win": edge_row["market_fair_away_win"],
        "home_odds": edge_row["home_odds"],
        "draw_odds": edge_row["draw_odds"],
        "away_odds": edge_row["away_odds"],
        "market_overround": edge_row["market_overround"],
        "edge_home_win": edge_row["edge_home_win"],
        "edge_draw": edge_row["edge_draw"],
        "edge_away_win": edge_row["edge_away_win"],
        "expected_value_home_win": edge_row["expected_value_home_win"],
        "expected_value_draw": edge_row["expected_value_draw"],
        "expected_value_away_win": edge_row["expected_value_away_win"],
        "best_outcome": edge_row["best_outcome"],
        "best_edge": edge_row["best_edge"],
        "best_expected_value": edge_row["best_expected_value"],
        "decision": edge_row["decision"],
        "ensemble_confidence": edge_row["ensemble_confidence"],
        "ensemble_entropy": edge_row["ensemble_entropy"],
        "max_model_disagreement": edge_row["max_model_disagreement"],
        "full_kelly_fraction": edge_row.get("full_kelly_fraction", ""),
        "suggested_stake_fraction": edge_row.get("suggested_stake_fraction", ""),
        "suggested_stake_amount": edge_row.get("suggested_stake_amount", ""),
        "stake_sizing_reason": edge_row.get("stake_sizing_reason", ""),
        "closing_home_odds": "",
        "closing_draw_odds": "",
        "closing_away_odds": "",
        "final_home_score": "",
        "final_away_score": "",
        "final_outcome": "",
        "realized_return": "",
        "notes": notes,
    }


def append_candidate_edges_to_prediction_ledger(
    batch_edges_path: str | Path,
    ledger_path: str | Path,
    model_version: str = "demo-v1",
    feature_version: str = "demo-features-v1",
    prediction_timestamp: str | None = None,
    notes: str = "",
) -> Path:
    """Append all candidate edges from a batch market edge file to the ledger."""

    batch_edges = pd.read_csv(batch_edges_path, keep_default_na=False)

    if "decision" not in batch_edges.columns:
        raise ValueError("Batch market edge file missing decision column.")

    if "strategy_action" in batch_edges.columns:
        candidate_edges = batch_edges[
            (batch_edges["decision"] == "candidate_edge")
            & (batch_edges["strategy_action"] == "actionable")
        ].copy()
    else:
        candidate_edges = batch_edges[
            batch_edges["decision"] == "candidate_edge"
        ].copy()

    destination = Path(ledger_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if candidate_edges.empty:
        if not destination.exists():
            pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(destination, index=False)
        return destination

    rows = [
        build_prediction_ledger_row_from_batch_edge(
            edge_row=row,
            model_version=model_version,
            feature_version=feature_version,
            prediction_timestamp=prediction_timestamp,
            notes=notes,
        )
        for _, row in candidate_edges.iterrows()
    ]

    new_rows = pd.DataFrame(rows, columns=LEDGER_COLUMNS)

    if destination.exists() and destination.stat().st_size > 0:
        existing = pd.read_csv(destination, keep_default_na=False, dtype=object)
        ledger = pd.concat([existing, new_rows], ignore_index=True)
    else:
        ledger = new_rows

    ledger.to_csv(destination, index=False)

    return destination


REQUIRED_SETTLEMENT_COLUMNS = [
    "home_team",
    "away_team",
    "final_home_score",
    "final_away_score",
]


def _ledger_has_value(value: object) -> bool:
    """Return whether a ledger CSV value is meaningfully populated."""

    return str(value).strip() != "" and str(value).strip().lower() != "nan"


def _optional_float_from_row(row: pd.Series, column: str) -> float | None:
    """Return an optional float from a settlement row."""

    if column not in row:
        return None

    value = row[column]

    if not _ledger_has_value(value):
        return None

    return float(value)


def validate_settlement_results(settlements: pd.DataFrame) -> None:
    """Validate batch settlement results input."""

    missing_columns = sorted(set(REQUIRED_SETTLEMENT_COLUMNS) - set(settlements.columns))

    if missing_columns:
        raise ValueError(
            f"Settlement results missing required columns: {missing_columns}"
        )

    if settlements.empty:
        raise ValueError("Settlement results file is empty.")

    for row_number, row in settlements.iterrows():
        home_team = str(row["home_team"]).strip()
        away_team = str(row["away_team"]).strip()

        if not home_team:
            raise ValueError(f"Row {row_number} has blank home_team.")

        if not away_team:
            raise ValueError(f"Row {row_number} has blank away_team.")

        if home_team == away_team:
            raise ValueError(f"Row {row_number} has identical teams: {home_team}.")

        int(row["final_home_score"])
        int(row["final_away_score"])


def settle_prediction_ledger_from_results(
    ledger_path: str | Path,
    settlement_results_path: str | Path,
    stake: float = 1.0,
) -> Path:
    """Settle all matching unsettled prediction ledger rows from a results CSV."""

    path = Path(ledger_path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction ledger not found: {path}")

    if stake <= 0.0:
        raise ValueError("stake must be positive.")

    ledger = pd.read_csv(path, keep_default_na=False, dtype=object)

    for column in LEDGER_COLUMNS:
        if column not in ledger.columns:
            ledger[column] = ""

    settlements = pd.read_csv(
        settlement_results_path,
        keep_default_na=False,
        dtype=object,
    )
    validate_settlement_results(settlements)

    for _, settlement in settlements.iterrows():
        home_team = str(settlement["home_team"]).strip()
        away_team = str(settlement["away_team"]).strip()
        kickoff_timestamp = str(settlement.get("kickoff_timestamp", "")).strip()

        final_home_score = int(settlement["final_home_score"])
        final_away_score = int(settlement["final_away_score"])
        final_outcome = match_outcome_from_score(
            home_score=final_home_score,
            away_score=final_away_score,
        )

        matching_rows = (
            (ledger["home_team"].astype(str).str.strip() == home_team)
            & (ledger["away_team"].astype(str).str.strip() == away_team)
            & ~ledger["final_outcome"].map(_ledger_has_value)
        )

        if kickoff_timestamp and "kickoff_timestamp" in ledger.columns:
            kickoff_matches = (
                ledger["kickoff_timestamp"].astype(str).str.strip()
                == kickoff_timestamp
            )
            matching_rows = matching_rows & (
                kickoff_matches
                | ~ledger["kickoff_timestamp"].map(_ledger_has_value)
            )

        closing_home_odds = _optional_float_from_row(
            row=settlement,
            column="closing_home_odds",
        )
        closing_draw_odds = _optional_float_from_row(
            row=settlement,
            column="closing_draw_odds",
        )
        closing_away_odds = _optional_float_from_row(
            row=settlement,
            column="closing_away_odds",
        )

        for row_index in ledger.index[matching_rows]:
            best_outcome = str(ledger.loc[row_index, "best_outcome"])

            row_stake = stake

            if (
                "suggested_stake_amount" in ledger.columns
                and _ledger_has_value(ledger.loc[row_index, "suggested_stake_amount"])
            ):
                row_stake = float(ledger.loc[row_index, "suggested_stake_amount"])

            realized_return = realized_return_for_prediction(
                best_outcome=best_outcome,
                final_outcome=final_outcome,
                home_odds=float(ledger.loc[row_index, "home_odds"]),
                draw_odds=float(ledger.loc[row_index, "draw_odds"]),
                away_odds=float(ledger.loc[row_index, "away_odds"]),
                stake=row_stake,
            )

            ledger.loc[row_index, "final_home_score"] = final_home_score
            ledger.loc[row_index, "final_away_score"] = final_away_score
            ledger.loc[row_index, "final_outcome"] = final_outcome
            ledger.loc[row_index, "realized_return"] = realized_return

            if closing_home_odds is not None:
                ledger.loc[row_index, "closing_home_odds"] = closing_home_odds

            if closing_draw_odds is not None:
                ledger.loc[row_index, "closing_draw_odds"] = closing_draw_odds

            if closing_away_odds is not None:
                ledger.loc[row_index, "closing_away_odds"] = closing_away_odds

    ledger.to_csv(path, index=False)

    return path
