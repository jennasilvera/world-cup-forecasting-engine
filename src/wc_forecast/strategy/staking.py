from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_STAKE_COLUMNS = [
    "home_team",
    "away_team",
    "strategy_action",
    "best_outcome",
    "home_odds",
    "draw_odds",
    "away_odds",
    "model_prob_home_win",
    "model_prob_draw",
    "model_prob_away_win",
]


@dataclass(frozen=True)
class StakeSizingPolicy:
    """Controls for converting actionable edges into suggested stake sizes."""

    bankroll: float = 1_000.0
    fractional_kelly: float = 0.25
    max_single_bet_fraction: float = 0.02
    max_portfolio_exposure_fraction: float = 0.05


def validate_stake_sizing_input(edges: pd.DataFrame) -> None:
    """Validate strategy-policy output before stake sizing."""

    missing_columns = sorted(set(REQUIRED_STAKE_COLUMNS) - set(edges.columns))

    if missing_columns:
        raise ValueError(
            f"Stake sizing input missing required columns: {missing_columns}"
        )

    if edges.empty:
        raise ValueError("Stake sizing input is empty.")


def validate_stake_sizing_policy(policy: StakeSizingPolicy) -> None:
    """Validate stake sizing configuration."""

    if policy.bankroll <= 0.0:
        raise ValueError("bankroll must be positive.")

    if not 0.0 < policy.fractional_kelly <= 1.0:
        raise ValueError("fractional_kelly must be between 0 and 1.")

    if not 0.0 < policy.max_single_bet_fraction <= 1.0:
        raise ValueError("max_single_bet_fraction must be between 0 and 1.")

    if not 0.0 < policy.max_portfolio_exposure_fraction <= 1.0:
        raise ValueError("max_portfolio_exposure_fraction must be between 0 and 1.")


def decimal_kelly_fraction(
    probability: float,
    decimal_odds: float,
) -> float:
    """Calculate full Kelly fraction for decimal odds."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")

    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.")

    net_odds = decimal_odds - 1.0
    loss_probability = 1.0 - probability

    return ((net_odds * probability) - loss_probability) / net_odds


def _best_outcome_probability(row: pd.Series) -> float:
    best_outcome = str(row["best_outcome"])

    if best_outcome == "home_win":
        return float(row["model_prob_home_win"])

    if best_outcome == "draw":
        return float(row["model_prob_draw"])

    if best_outcome == "away_win":
        return float(row["model_prob_away_win"])

    raise ValueError(f"Unsupported best_outcome: {best_outcome}")


def _best_outcome_odds(row: pd.Series) -> float:
    best_outcome = str(row["best_outcome"])

    if best_outcome == "home_win":
        return float(row["home_odds"])

    if best_outcome == "draw":
        return float(row["draw_odds"])

    if best_outcome == "away_win":
        return float(row["away_odds"])

    raise ValueError(f"Unsupported best_outcome: {best_outcome}")


def apply_stake_sizing(
    edges: pd.DataFrame,
    policy: StakeSizingPolicy | None = None,
) -> pd.DataFrame:
    """Apply fractional Kelly sizing with single-bet and portfolio caps."""

    validate_stake_sizing_input(edges)

    active_policy = policy or StakeSizingPolicy()
    validate_stake_sizing_policy(active_policy)

    output = edges.copy()

    full_kelly_fractions: list[float] = []
    suggested_fractions: list[float] = []
    reasons: list[str] = []

    for _, row in output.iterrows():
        if str(row["strategy_action"]) != "actionable":
            full_kelly_fractions.append(0.0)
            suggested_fractions.append(0.0)
            reasons.append("not_actionable")
            continue

        probability = _best_outcome_probability(row)
        decimal_odds = _best_outcome_odds(row)
        full_kelly = decimal_kelly_fraction(
            probability=probability,
            decimal_odds=decimal_odds,
        )

        if full_kelly <= 0.0:
            full_kelly_fractions.append(full_kelly)
            suggested_fractions.append(0.0)
            reasons.append("non_positive_kelly")
            continue

        fractional_kelly = full_kelly * active_policy.fractional_kelly
        capped_fraction = min(
            fractional_kelly,
            active_policy.max_single_bet_fraction,
        )

        full_kelly_fractions.append(full_kelly)
        suggested_fractions.append(capped_fraction)

        if capped_fraction < fractional_kelly:
            reasons.append("single_bet_cap_applied")
        else:
            reasons.append("fractional_kelly")

    total_fraction = sum(suggested_fractions)
    scale_factor = 1.0

    if total_fraction > active_policy.max_portfolio_exposure_fraction:
        scale_factor = active_policy.max_portfolio_exposure_fraction / total_fraction
        suggested_fractions = [
            fraction * scale_factor for fraction in suggested_fractions
        ]
        reasons = [
            f"{reason};portfolio_cap_scaled"
            if fraction > 0.0
            else reason
            for reason, fraction in zip(reasons, suggested_fractions, strict=True)
        ]

    output["full_kelly_fraction"] = full_kelly_fractions
    output["suggested_stake_fraction"] = suggested_fractions
    output["suggested_stake_amount"] = [
        fraction * active_policy.bankroll for fraction in suggested_fractions
    ]
    output["stake_sizing_reason"] = reasons

    output = output.sort_values(
        by=["suggested_stake_amount", "best_expected_value"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return output


def save_stake_sizing_output(
    strategy_policy_path: str | Path,
    output_path: str | Path,
    policy: StakeSizingPolicy | None = None,
) -> Path:
    """Apply stake sizing to a strategy-policy CSV and save the output."""

    edges = pd.read_csv(strategy_policy_path)
    sized = apply_stake_sizing(edges=edges, policy=policy)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sized.to_csv(destination, index=False)

    return destination
