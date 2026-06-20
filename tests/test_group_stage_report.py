from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.reporting.group_stage_report import (
    render_group_stage_report,
    save_group_stage_report,
    validate_group_stage_summary,
)


def _sample_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["Group A", "Group A", "Group B", "Group B"],
            "team": ["Argentina", "France", "England", "Netherlands"],
            "simulations": [100, 100, 100, 100],
            "advance_probability": [0.70, 0.55, 0.80, 0.45],
            "avg_points": [5.8, 4.9, 6.2, 4.4],
            "avg_goal_difference": [1.4, 0.7, 2.1, 0.3],
            "avg_goals_for": [5.2, 4.7, 6.1, 4.1],
        }
    )


def test_render_group_stage_report_contains_group_sections() -> None:
    report = render_group_stage_report(_sample_summary())

    assert "# Group-Stage Simulation Report" in report
    assert "### Group A" in report
    assert "Argentina" in report
    assert "Advance Probability" in report


def test_validate_group_stage_summary_rejects_missing_columns() -> None:
    summary = _sample_summary().drop(columns=["advance_probability"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_group_stage_summary(summary)


def test_validate_group_stage_summary_rejects_invalid_probability() -> None:
    summary = _sample_summary()
    summary.loc[0, "advance_probability"] = 1.25

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_group_stage_summary(summary)


def test_save_group_stage_report_writes_markdown(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    report_path = tmp_path / "report.md"

    _sample_summary().to_csv(summary_path, index=False)

    destination = save_group_stage_report(
        summary_path=summary_path,
        output_path=report_path,
    )

    assert destination.exists()
    assert "Group-Stage Simulation Report" in report_path.read_text()
