from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.reporting.prediction_report import (
    render_backtest_report,
    save_backtest_report,
    validate_backtest_inputs,
)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2022-12-18"],
            "home_team": ["Argentina"],
            "away_team": ["France"],
            "tournament": ["FIFA World Cup"],
            "home_score": [3],
            "away_score": [3],
            "actual_outcome": ["draw"],
            "predicted_outcome": ["home_win"],
            "prob_home_win": [0.45],
            "prob_draw": [0.30],
            "prob_away_win": [0.25],
        }
    )


def _sample_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "train_rows",
                "test_rows",
                "accuracy",
                "log_loss",
                "multiclass_brier_score",
            ],
            "value": [7.0, 3.0, 0.3333, 1.2615, 0.6231],
        }
    )


def test_render_backtest_report_contains_model_summary() -> None:
    report = render_backtest_report(
        predictions=_sample_predictions(),
        metrics=_sample_metrics(),
    )

    assert "# Backtest Report" in report
    assert "Logistic Regression Baseline" in report
    assert "Argentina vs France" in report
    assert "multiclass Brier score" in report


def test_validate_backtest_inputs_rejects_missing_probability_column() -> None:
    predictions = _sample_predictions().drop(columns=["prob_draw"])

    with pytest.raises(ValueError, match="Prediction table missing required columns"):
        validate_backtest_inputs(predictions, _sample_metrics())


def test_validate_backtest_inputs_rejects_probabilities_not_summing_to_one() -> None:
    predictions = _sample_predictions()
    predictions.loc[0, "prob_home_win"] = 0.80

    with pytest.raises(ValueError, match="Prediction probabilities must sum to 1.0"):
        validate_backtest_inputs(predictions, _sample_metrics())


def test_save_backtest_report_writes_markdown_file(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"
    report_path = tmp_path / "report.md"

    _sample_predictions().to_csv(predictions_path, index=False)
    _sample_metrics().to_csv(metrics_path, index=False)

    destination = save_backtest_report(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        output_path=report_path,
    )

    assert destination == report_path
    assert report_path.exists()
    assert "Backtest Report" in report_path.read_text()
