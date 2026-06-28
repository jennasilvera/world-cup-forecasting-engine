from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.reports.closing_line_value import decimal_implied_probability

REQUIRED_MARKET_SNAPSHOT_COLUMNS = {
    "fixture_id",
    "snapshot_timestamp",
    "home_team",
    "away_team",
    "market_home_odds",
    "market_draw_odds",
    "market_away_odds",
}

OUTCOME_ODDS_COLUMNS = {
    "home_win": "market_home_odds",
    "draw": "market_draw_odds",
    "away_win": "market_away_odds",
}


def validate_market_snapshots_frame(frame: pd.DataFrame) -> None:
    """Validate market snapshot input columns."""

    missing = REQUIRED_MARKET_SNAPSHOT_COLUMNS - set(frame.columns)

    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing market snapshot columns: {missing_columns}")


def build_market_movement_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Build market movement analytics from timestamped odds snapshots."""

    validate_market_snapshots_frame(snapshots)

    market = snapshots.copy()
    market["snapshot_timestamp"] = pd.to_datetime(
        market["snapshot_timestamp"],
        utc=True,
        errors="coerce",
    )

    if market["snapshot_timestamp"].isna().any():
        raise ValueError("snapshot_timestamp contains invalid timestamps")

    for column in OUTCOME_ODDS_COLUMNS.values():
        market[column] = pd.to_numeric(market[column], errors="coerce")

    if market[list(OUTCOME_ODDS_COLUMNS.values())].isna().any().any():
        raise ValueError("Market odds columns must contain numeric decimal odds")

    market = market.sort_values(["fixture_id", "snapshot_timestamp"])

    opening = market.groupby("fixture_id", as_index=False).first()
    latest = market.groupby("fixture_id", as_index=False).last()

    rows: list[dict[str, object]] = []

    for _, latest_row in latest.iterrows():
        fixture_id = latest_row["fixture_id"]
        opening_row = opening.loc[opening["fixture_id"] == fixture_id].iloc[0]

        row: dict[str, object] = {
            "fixture_id": fixture_id,
            "home_team": latest_row["home_team"],
            "away_team": latest_row["away_team"],
            "opening_timestamp": opening_row["snapshot_timestamp"].isoformat(),
            "latest_timestamp": latest_row["snapshot_timestamp"].isoformat(),
            "snapshot_count": int(
                len(market.loc[market["fixture_id"] == fixture_id])
            ),
        }

        movement_values: dict[str, float] = {}

        for outcome, odds_column in OUTCOME_ODDS_COLUMNS.items():
            opening_odds = float(opening_row[odds_column])
            latest_odds = float(latest_row[odds_column])

            opening_implied = decimal_implied_probability(opening_odds)
            latest_implied = decimal_implied_probability(latest_odds)
            implied_probability_move = latest_implied - opening_implied
            odds_move = latest_odds - opening_odds

            row[f"opening_{outcome}_odds"] = opening_odds
            row[f"latest_{outcome}_odds"] = latest_odds
            row[f"opening_{outcome}_implied_probability"] = opening_implied
            row[f"latest_{outcome}_implied_probability"] = latest_implied
            row[f"{outcome}_odds_move"] = odds_move
            row[f"{outcome}_implied_probability_move"] = implied_probability_move

            movement_values[outcome] = implied_probability_move

        largest_outcome = max(
            movement_values,
            key=lambda outcome: abs(movement_values[outcome]),
        )

        row["largest_market_move_outcome"] = largest_outcome
        row["largest_market_move"] = movement_values[largest_outcome]
        row["market_direction"] = _market_direction(
            largest_outcome,
            movement_values[largest_outcome],
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("fixture_id").reset_index(drop=True)


def save_market_movement_table(
    snapshots_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Read market snapshots and write market movement analytics."""

    snapshots = pd.read_csv(snapshots_path)
    movement = build_market_movement_table(snapshots)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    movement.to_csv(destination, index=False)

    return movement


def _market_direction(outcome: str, implied_probability_move: float) -> str:
    if implied_probability_move > 0:
        return f"market_toward_{outcome}"

    if implied_probability_move < 0:
        return f"market_away_from_{outcome}"

    return "market_unchanged"
