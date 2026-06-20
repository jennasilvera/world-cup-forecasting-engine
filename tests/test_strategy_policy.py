from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.strategy.policy import (
    StrategyPolicy,
    apply_strategy_policy,
    save_strategy_policy_output,
    validate_strategy_policy_input,
)


def _sample_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["Brazil", "Argentina", "England"],
            "away_team": ["Spain", "France", "Ecuador"],
            "decision": ["candidate_edge", "candidate_edge", "candidate_edge"],
            "best_outcome": ["draw", "draw", "home_win"],
            "best_edge": [0.445, 0.247, 0.104],
            "best_expected_value": [1.427, 0.806, 0.161],
            "market_overround": [1.045, 1.034, 1.042],
            "ensemble_confidence": ["High", "Medium", "Medium"],
            "ensemble_entropy": [0.683, 0.915, 0.859],
            "max_model_disagreement": [0.111, 0.777, 0.420],
        }
    )


def test_validate_strategy_policy_input_rejects_missing_columns() -> None:
    edges = _sample_edges().drop(columns=["best_edge"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_strategy_policy_input(edges)


def test_apply_strategy_policy_marks_actionable_and_filtered() -> None:
    output = apply_strategy_policy(_sample_edges())

    assert "strategy_action" in output.columns
    assert "strategy_reason" in output.columns
    assert set(output["strategy_action"]) == {"actionable", "filtered"}


def test_apply_strategy_policy_can_be_tightened() -> None:
    policy = StrategyPolicy(
        minimum_edge=0.50,
        minimum_expected_value=1.50,
    )

    output = apply_strategy_policy(
        edges=_sample_edges(),
        policy=policy,
    )

    assert set(output["strategy_action"]) == {"filtered"}


def test_save_strategy_policy_output_writes_csv(tmp_path) -> None:
    batch_edges_path = tmp_path / "batch_market_edges.csv"
    output_path = tmp_path / "strategy_policy_edges.csv"

    _sample_edges().to_csv(batch_edges_path, index=False)

    destination = save_strategy_policy_output(
        batch_edges_path=batch_edges_path,
        output_path=output_path,
    )

    saved = pd.read_csv(destination)

    assert destination.exists()
    assert "strategy_action" in saved.columns
    assert len(saved) == 3
