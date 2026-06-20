from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.reporting.ledger_report import (
    render_prediction_ledger_report,
    save_prediction_ledger_report,
    summarize_prediction_ledger,
    validate_prediction_ledger_for_report,
)


def _sample_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_id": ["abc123", "def456"],
            "home_team": ["Argentina", "Brazil"],
            "away_team": ["France", "Spain"],
            "decision": ["candidate_edge", "no_edge"],
            "best_outcome": ["draw", "home_win"],
            "best_edge": [0.10, 0.01],
            "best_expected_value": [0.25, -0.02],
            "home_odds": [2.20, 1.90],
            "draw_odds": [3.40, 3.20],
            "away_odds": [3.50, 4.00],
            "closing_home_odds": [2.10, ""],
            "closing_draw_odds": [3.25, ""],
            "closing_away_odds": [3.60, ""],
            "final_outcome": ["draw", ""],
            "realized_return": [2.40, ""],
        }
    )


def test_summarize_prediction_ledger_returns_performance_metrics() -> None:
    summary = summarize_prediction_ledger(_sample_ledger())

    assert summary["total_predictions"] == 2
    assert summary["settled_predictions"] == 1
    assert summary["candidate_edges_logged"] == 1
    assert summary["hit_rate"] == 1.0
    assert summary["total_realized_return"] == 2.40


def test_render_prediction_ledger_report_contains_sections() -> None:
    report = render_prediction_ledger_report(_sample_ledger())

    assert "# Prediction Ledger Performance Report" in report
    assert "Summary Metrics" in report
    assert "Recent Settled Predictions" in report
    assert "Argentina vs France" in report


def test_validate_prediction_ledger_for_report_rejects_missing_columns() -> None:
    ledger = _sample_ledger().drop(columns=["realized_return"])

    with pytest.raises(ValueError, match="missing required report columns"):
        validate_prediction_ledger_for_report(ledger)


def test_save_prediction_ledger_report_writes_markdown(tmp_path) -> None:
    ledger_path = tmp_path / "prediction_ledger.csv"
    report_path = tmp_path / "ledger_report.md"

    _sample_ledger().to_csv(ledger_path, index=False)

    destination = save_prediction_ledger_report(
        ledger_path=ledger_path,
        output_path=report_path,
    )

    assert destination.exists()
    assert "Prediction Ledger Performance Report" in report_path.read_text()


def test_summarize_prediction_ledger_includes_stake_weighted_metrics() -> None:
    ledger = _sample_ledger()
    ledger["suggested_stake_amount"] = [20.0, 10.0]

    summary = summarize_prediction_ledger(ledger)

    assert "stake_weighted_return" in summary
    assert "stake_weighted_roi" in summary
    assert "total_suggested_exposure" in summary
    assert "average_suggested_stake" in summary
    assert summary["total_suggested_exposure"] > 0


def test_render_prediction_ledger_report_includes_stake_weighted_metrics() -> None:
    ledger = _sample_ledger()
    ledger["suggested_stake_amount"] = [20.0, 10.0]

    report = render_prediction_ledger_report(ledger)

    assert "Stake-weighted return" in report
    assert "Stake-weighted ROI" in report
    assert "Total suggested exposure" in report
    assert "Average suggested stake" in report
    expected_header = (
        "| Prediction ID | Match | Decision | Pick | Final | Stake | Return | EV | Edge |"
    )
    assert expected_header in report
