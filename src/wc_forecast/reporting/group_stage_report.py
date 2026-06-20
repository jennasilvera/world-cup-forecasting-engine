from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_GROUP_SUMMARY_COLUMNS = [
    "group",
    "team",
    "simulations",
    "advance_probability",
    "avg_points",
    "avg_goal_difference",
    "avg_goals_for",
]


def _format_probability(value: float) -> str:
    """Format a probability as a percentage string."""

    return f"{value * 100:.1f}%"


def validate_group_stage_summary(summary: pd.DataFrame) -> None:
    """Validate group-stage simulation summary output before reporting."""

    missing_columns = sorted(
        set(REQUIRED_GROUP_SUMMARY_COLUMNS) - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Group-stage summary missing required columns: "
            f"{missing_columns}"
        )

    if summary.empty:
        raise ValueError("Group-stage summary is empty.")

    if summary["advance_probability"].isna().any():
        raise ValueError("Advance probability contains missing values.")

    if not summary["advance_probability"].between(0.0, 1.0).all():
        raise ValueError("Advance probabilities must be between 0 and 1.")


def render_group_stage_report(
    summary: pd.DataFrame,
    title: str = "Group-Stage Simulation Report",
) -> str:
    """Render a Markdown report from group-stage simulation summary data."""

    validate_group_stage_summary(summary)

    sorted_summary = summary.sort_values(
        ["group", "advance_probability", "avg_points", "avg_goal_difference"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    simulation_count = int(sorted_summary["simulations"].max())
    groups = sorted(sorted_summary["group"].unique())

    sections: list[str] = []

    for group_name in groups:
        group_rows = sorted_summary[sorted_summary["group"] == group_name]

        table_rows = [
            "| Team | Advance Probability | Avg Points | Avg GD | Avg GF |",
            "|---|---:|---:|---:|---:|",
        ]

        for row in group_rows.itertuples(index=False):
            table_rows.append(
                "| "
                f"{row.team} | "
                f"{_format_probability(float(row.advance_probability))} | "
                f"{float(row.avg_points):.2f} | "
                f"{float(row.avg_goal_difference):.2f} | "
                f"{float(row.avg_goals_for):.2f} |"
            )

        sections.append(f"### {group_name}\n\n" + "\n".join(table_rows))

    return f"""# {title}

## Purpose

This report summarizes a Monte Carlo group-stage simulation for the World Cup
Match Forecasting Engine.

The simulation repeatedly samples match scorelines from the Poisson
expected-goals model, builds group standings, applies ranking logic, and
estimates each team's probability of advancing from its group.

## Simulation Setup

| Field | Value |
|---|---:|
| Simulations per team/group estimate | {simulation_count} |
| Ranking logic | Points, goal difference, goals for, team name |
| Match scoring model | Poisson expected-goals baseline |
| Advancement rule | Top teams by simulated group rank |

## Advancement Summary

{chr(10).join(sections)}

## Interpretation

Advance probability estimates represent the share of simulations in which a team
finished in an advancing position.

Average points, average goal difference, and average goals for help explain
whether a team is advancing through consistent simulated performance or through
thin margins.

## Caveats

- The current committed fixture file is a small reproducible demo sample.
- The current historical results dataset is intentionally small.
- These probabilities are pipeline outputs, not validated betting or trading
  signals.
- FIFA tie-breaker logic is simplified.
- Knockout-stage simulation is not implemented yet.
- Future versions should use a larger historical dataset, calibrated model
  weights, rest/travel context, market odds, squad strength, and player
  availability inputs.
"""


def save_group_stage_report(
    summary_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load group-stage simulation summary CSV and save Markdown report."""

    summary = pd.read_csv(summary_path)
    report = render_group_stage_report(summary)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report)

    return destination
