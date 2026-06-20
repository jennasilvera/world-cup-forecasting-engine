from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.strategy.staking import (
    StakeSizingPolicy,
    apply_stake_sizing,
    decimal_kelly_fraction,
    save_stake_sizing_output,
    validate_stake_sizing_input,
)


def _sample_strategy_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["Brazil", "England", "Argentina"],
            "away_team": ["Spain", "Ecuador", "France"],
            "strategy_action": ["actionable", "actionable", "filtered"],
            "strategy_reason": [
                "passes_policy",
                "passes_policy",
                "model_disagreement_above_limit",
            ],
            "best_outcome": ["draw", "home_win", "draw"],
            "best_expected_value": [1.427, 0.161, 0.806],
            "home_odds": [2.60, 1.95, 2.20],
            "draw_odds": [3.30, 3.50, 3.40],
            "away_odds": [2.80, 4.10, 3.50],
            "model_prob_home_win": [0.169, 0.595, 0.183],
            "model_prob_draw": [0.736, 0.153, 0.531],
            "model_prob_away_win": [0.096, 0.251, 0.286],
        }
    )


def test_decimal_kelly_fraction_positive_edge() -> None:
    fraction = decimal_kelly_fraction(
        probability=0.60,
        decimal_odds=2.00,
    )

    assert fraction == pytest.approx(0.20)


def test_validate_stake_sizing_input_rejects_missing_columns() -> None:
    edges = _sample_strategy_edges().drop(columns=["best_outcome"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_stake_sizing_input(edges)


def test_apply_stake_sizing_sets_filtered_rows_to_zero() -> None:
    sized = apply_stake_sizing(_sample_strategy_edges())

    filtered = sized[sized["strategy_action"] == "filtered"].iloc[0]

    assert filtered["suggested_stake_fraction"] == 0.0
    assert filtered["suggested_stake_amount"] == 0.0
    assert filtered["stake_sizing_reason"] == "not_actionable"


def test_apply_stake_sizing_applies_single_bet_cap() -> None:
    policy = StakeSizingPolicy(
        bankroll=1_000.0,
        fractional_kelly=0.25,
        max_single_bet_fraction=0.02,
        max_portfolio_exposure_fraction=0.10,
    )

    sized = apply_stake_sizing(
        edges=_sample_strategy_edges(),
        policy=policy,
    )

    actionable = sized[sized["strategy_action"] == "actionable"]

    assert actionable["suggested_stake_fraction"].max() <= 0.02
    assert actionable["suggested_stake_amount"].max() <= 20.0


def test_apply_stake_sizing_applies_portfolio_cap() -> None:
    policy = StakeSizingPolicy(
        bankroll=1_000.0,
        fractional_kelly=0.25,
        max_single_bet_fraction=0.05,
        max_portfolio_exposure_fraction=0.03,
    )

    sized = apply_stake_sizing(
        edges=_sample_strategy_edges(),
        policy=policy,
    )

    assert sized["suggested_stake_fraction"].sum() <= 0.03 + 1e-12


def test_save_stake_sizing_output_writes_csv(tmp_path) -> None:
    strategy_policy_path = tmp_path / "strategy_policy_edges.csv"
    output_path = tmp_path / "stake_sizing_edges.csv"

    _sample_strategy_edges().to_csv(strategy_policy_path, index=False)

    destination = save_stake_sizing_output(
        strategy_policy_path=strategy_policy_path,
        output_path=output_path,
    )

    saved = pd.read_csv(destination)

    assert destination.exists()
    assert "suggested_stake_amount" in saved.columns
    assert len(saved) == 3
