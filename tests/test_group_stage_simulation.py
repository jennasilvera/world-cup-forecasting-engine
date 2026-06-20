from __future__ import annotations

import pandas as pd
import pytest

from wc_forecast.simulation.group_stage import (
    load_group_fixtures,
    save_group_stage_simulation,
    simulate_group_stage,
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


def _sample_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["Group A", "Group A", "Group A"],
            "home_team": ["Argentina", "Argentina", "France"],
            "away_team": ["France", "Brazil", "Brazil"],
            "neutral": [True, True, True],
        }
    )


def test_simulate_group_stage_outputs_advancement_summary() -> None:
    result = simulate_group_stage(
        results=_sample_results(),
        fixtures=_sample_fixtures(),
        n_simulations=20,
        seed=7,
    )

    assert {"group", "team", "advance_probability"}.issubset(result.summary.columns)
    assert result.summary["advance_probability"].between(0.0, 1.0).all()
    assert result.standings["simulation_id"].nunique() == 20


def test_load_group_fixtures_rejects_missing_columns(tmp_path) -> None:
    fixtures_path = tmp_path / "bad_fixtures.csv"
    fixtures_path.write_text("group,home_team,away_team\nGroup A,Argentina,France\n")

    with pytest.raises(ValueError, match="Fixture table missing required columns"):
        load_group_fixtures(fixtures_path)


def test_save_group_stage_simulation_writes_summary(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    fixtures_path = tmp_path / "fixtures.csv"
    output_path = tmp_path / "simulation.csv"

    _sample_results().to_csv(results_path, index=False)
    _sample_fixtures().to_csv(fixtures_path, index=False)

    destination = save_group_stage_simulation(
        results_path=results_path,
        fixtures_path=fixtures_path,
        output_path=output_path,
        n_simulations=10,
        seed=42,
    )

    assert destination.exists()
    saved = pd.read_csv(destination)
    assert "advance_probability" in saved.columns
