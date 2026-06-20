from __future__ import annotations

import pandas as pd

from wc_forecast.reporting.match_report import (
    build_current_match_features,
    generate_match_prediction,
    render_match_prediction_report,
    save_match_prediction_report,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-11-20",
                    "2022-11-21",
                    "2022-11-21",
                    "2022-11-22",
                    "2022-11-22",
                    "2022-11-23",
                    "2022-11-24",
                    "2022-11-25",
                    "2022-12-03",
                    "2022-12-18",
                ]
            ),
            "home_team": [
                "Qatar",
                "England",
                "United States",
                "Argentina",
                "Mexico",
                "Spain",
                "Brazil",
                "Netherlands",
                "Netherlands",
                "Argentina",
            ],
            "away_team": [
                "Ecuador",
                "Iran",
                "Wales",
                "Saudi Arabia",
                "Poland",
                "Costa Rica",
                "Serbia",
                "Ecuador",
                "United States",
                "France",
            ],
            "home_score": [0, 6, 1, 1, 0, 7, 2, 1, 3, 3],
            "away_score": [2, 2, 1, 2, 0, 0, 0, 1, 1, 3],
            "tournament": ["FIFA World Cup"] * 10,
            "city": [
                "Al Khor",
                "Al Rayyan",
                "Al Rayyan",
                "Lusail",
                "Doha",
                "Doha",
                "Lusail",
                "Doha",
                "Khalifa",
                "Lusail",
            ],
            "country": ["Qatar"] * 10,
            "neutral": [False, True, True, True, True, True, True, True, True, True],
            "outcome": [
                "away_win",
                "home_win",
                "draw",
                "away_win",
                "draw",
                "home_win",
                "home_win",
                "draw",
                "home_win",
                "draw",
            ],
        }
    )


def test_build_current_match_features_has_one_row() -> None:
    features = build_current_match_features(
        results=_sample_results(),
        home_team="Argentina",
        away_team="France",
    )

    assert len(features) == 1
    assert "elo_diff_home_minus_away" in features.columns


def test_generate_match_prediction_returns_probability_layers() -> None:
    prediction = generate_match_prediction(
        results=_sample_results(),
        home_team="Argentina",
        away_team="France",
    )

    logistic_sum = (
        float(prediction["logistic_prob_home_win"])
        + float(prediction["logistic_prob_draw"])
        + float(prediction["logistic_prob_away_win"])
    )

    poisson_sum = (
        float(prediction["poisson_prob_home_win"])
        + float(prediction["poisson_prob_draw"])
        + float(prediction["poisson_prob_away_win"])
    )

    assert round(logistic_sum, 8) == 1.0
    assert round(poisson_sum, 8) == 1.0
    assert "most_likely_score" in prediction


def test_render_match_prediction_report_contains_match_context() -> None:
    prediction = generate_match_prediction(
        results=_sample_results(),
        home_team="Argentina",
        away_team="France",
    )

    report = render_match_prediction_report(prediction)

    assert "Match Prediction Report: Argentina vs France" in report
    assert "Logistic Regression Baseline" in report
    assert "Poisson Expected-Goals Forecast" in report


def test_save_match_prediction_report_writes_outputs(tmp_path) -> None:
    raw_path = tmp_path / "results.csv"
    prediction_path = tmp_path / "prediction.csv"
    report_path = tmp_path / "report.md"

    _sample_results().to_csv(raw_path, index=False)

    prediction_destination, report_destination = save_match_prediction_report(
        results_path=raw_path,
        home_team="Argentina",
        away_team="France",
        prediction_output_path=prediction_path,
        report_output_path=report_path,
    )

    assert prediction_destination.exists()
    assert report_destination.exists()
    assert "Argentina vs France" in report_path.read_text()
