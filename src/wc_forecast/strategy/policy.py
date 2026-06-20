from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_STRATEGY_COLUMNS = [
    "home_team",
    "away_team",
    "decision",
    "best_outcome",
    "best_edge",
    "best_expected_value",
    "market_overround",
    "ensemble_confidence",
    "ensemble_entropy",
    "max_model_disagreement",
]


@dataclass(frozen=True)
class StrategyPolicy:
    """Risk and quality gates for turning candidate edges into actions."""

    minimum_edge: float = 0.05
    minimum_expected_value: float = 0.05
    maximum_entropy: float = 1.00
    maximum_model_disagreement: float = 0.50
    maximum_market_overround: float = 1.08
    allowed_confidences: tuple[str, ...] = ("High", "Medium")


def _to_float(value: object) -> float:
    """Convert a CSV value to float."""

    return float(value)


def validate_strategy_policy_input(edges: pd.DataFrame) -> None:
    """Validate batch edge data before applying policy rules."""

    missing_columns = sorted(set(REQUIRED_STRATEGY_COLUMNS) - set(edges.columns))

    if missing_columns:
        raise ValueError(
            f"Strategy policy input missing required columns: {missing_columns}"
        )

    if edges.empty:
        raise ValueError("Strategy policy input is empty.")


def explain_policy_rejection(
    row: pd.Series,
    policy: StrategyPolicy,
) -> list[str]:
    """Return policy rejection reasons for one edge row."""

    reasons: list[str] = []

    if str(row["decision"]) != "candidate_edge":
        reasons.append("decision_not_candidate_edge")

    if _to_float(row["best_edge"]) < policy.minimum_edge:
        reasons.append("edge_below_threshold")

    if _to_float(row["best_expected_value"]) < policy.minimum_expected_value:
        reasons.append("expected_value_below_threshold")

    if _to_float(row["ensemble_entropy"]) > policy.maximum_entropy:
        reasons.append("entropy_above_limit")

    if _to_float(row["max_model_disagreement"]) > policy.maximum_model_disagreement:
        reasons.append("model_disagreement_above_limit")

    if _to_float(row["market_overround"]) > policy.maximum_market_overround:
        reasons.append("market_overround_above_limit")

    if str(row["ensemble_confidence"]) not in policy.allowed_confidences:
        reasons.append("confidence_not_allowed")

    return reasons


def apply_strategy_policy(
    edges: pd.DataFrame,
    policy: StrategyPolicy | None = None,
) -> pd.DataFrame:
    """Apply strategy policy rules to a batch market edge table."""

    validate_strategy_policy_input(edges)

    active_policy = policy or StrategyPolicy()
    output = edges.copy()

    actions: list[str] = []
    reasons: list[str] = []

    for _, row in output.iterrows():
        rejection_reasons = explain_policy_rejection(
            row=row,
            policy=active_policy,
        )

        if rejection_reasons:
            actions.append("filtered")
            reasons.append(";".join(rejection_reasons))
        else:
            actions.append("actionable")
            reasons.append("passes_policy")

    output["strategy_action"] = actions
    output["strategy_reason"] = reasons

    action_rank = {"actionable": 0, "filtered": 1}
    output["_strategy_action_rank"] = output["strategy_action"].map(action_rank)
    output = output.sort_values(
        by=["_strategy_action_rank", "best_expected_value", "best_edge"],
        ascending=[True, False, False],
    )
    output = output.drop(columns=["_strategy_action_rank"]).reset_index(drop=True)

    return output


def save_strategy_policy_output(
    batch_edges_path: str | Path,
    output_path: str | Path,
    policy: StrategyPolicy | None = None,
) -> Path:
    """Apply strategy policy to a batch edge CSV and save the result."""

    edges = pd.read_csv(batch_edges_path)
    policy_output = apply_strategy_policy(
        edges=edges,
        policy=policy,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    policy_output.to_csv(destination, index=False)

    return destination
