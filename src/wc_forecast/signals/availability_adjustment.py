from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.signals.player_availability import build_fixture_availability_features

PROBABILITY_COLUMNS = ["prob_home_win", "prob_draw", "prob_away_win"]


def apply_availability_adjustment(
    forecasts: pd.DataFrame,
    team_availability_features: pd.DataFrame,
    adjustment_strength: float = 0.10,
    max_shift: float = 0.08,
) -> pd.DataFrame:
    """Apply player-availability signal adjustments to forecast probabilities."""

    _validate_forecasts(forecasts)

    enriched = build_fixture_availability_features(
        fixtures=forecasts,
        team_features=team_availability_features,
    )

    availability_edge = enriched["availability_pct_diff"].fillna(0.0)
    unavailable_value_edge = (
        enriched["away_unavailable_value_pct"].fillna(0.0)
        - enriched["home_unavailable_value_pct"].fillna(0.0)
    )

    raw_shift = adjustment_strength * availability_edge + 0.50 * adjustment_strength * (
        unavailable_value_edge
    )
    bounded_shift = raw_shift.clip(lower=-max_shift, upper=max_shift)

    adjusted = enriched.copy()
    adjusted["availability_probability_shift"] = bounded_shift

    adjusted["adjusted_prob_home_win"] = (
        adjusted["prob_home_win"] + bounded_shift
    ).clip(lower=0.01)
    adjusted["adjusted_prob_draw"] = adjusted["prob_draw"].clip(lower=0.01)
    adjusted["adjusted_prob_away_win"] = (
        adjusted["prob_away_win"] - bounded_shift
    ).clip(lower=0.01)

    total = (
        adjusted["adjusted_prob_home_win"]
        + adjusted["adjusted_prob_draw"]
        + adjusted["adjusted_prob_away_win"]
    )

    adjusted["adjusted_prob_home_win"] = adjusted["adjusted_prob_home_win"] / total
    adjusted["adjusted_prob_draw"] = adjusted["adjusted_prob_draw"] / total
    adjusted["adjusted_prob_away_win"] = adjusted["adjusted_prob_away_win"] / total

    adjusted["adjusted_predicted_outcome"] = adjusted.apply(
        _adjusted_outcome,
        axis=1,
    )
    adjusted["adjusted_predicted_winner"] = adjusted.apply(
        _adjusted_winner,
        axis=1,
    )
    adjusted["adjusted_model_confidence"] = adjusted[
        [
            "adjusted_prob_home_win",
            "adjusted_prob_draw",
            "adjusted_prob_away_win",
        ]
    ].max(axis=1)

    adjusted["availability_adjustment_applied"] = (
        adjusted["availability_probability_shift"].abs() > 0
    )

    return adjusted


def save_availability_adjusted_forecasts(
    forecasts_path: str | Path,
    availability_features_path: str | Path,
    output_path: str | Path,
    adjustment_strength: float = 0.10,
    max_shift: float = 0.08,
) -> pd.DataFrame:
    """Read forecasts and availability features, then write adjusted forecasts."""

    forecasts = pd.read_csv(forecasts_path)
    availability_features = pd.read_csv(availability_features_path)

    adjusted = apply_availability_adjustment(
        forecasts=forecasts,
        team_availability_features=availability_features,
        adjustment_strength=adjustment_strength,
        max_shift=max_shift,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(destination, index=False)

    return adjusted


def _validate_forecasts(forecasts: pd.DataFrame) -> None:
    required = {"home_team", "away_team", *PROBABILITY_COLUMNS}
    missing = required - set(forecasts.columns)

    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing forecast columns: {missing_columns}")


def _adjusted_outcome(row: pd.Series) -> str:
    probabilities = {
        "home_win": row["adjusted_prob_home_win"],
        "draw": row["adjusted_prob_draw"],
        "away_win": row["adjusted_prob_away_win"],
    }

    return max(probabilities, key=probabilities.get)


def _adjusted_winner(row: pd.Series) -> str:
    outcome = row["adjusted_predicted_outcome"]

    if outcome == "home_win":
        return str(row["home_team"])

    if outcome == "away_win":
        return str(row["away_team"])

    return "Draw"
