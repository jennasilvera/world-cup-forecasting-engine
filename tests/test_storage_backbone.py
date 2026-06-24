from __future__ import annotations

import sqlite3

import pandas as pd

from wc_forecast.storage.database import initialize_database, table_names
from wc_forecast.storage.external_signals import (
    write_injury_suspension_events,
    write_market_snapshots,
    write_player_availability,
)
from wc_forecast.storage.feature_store import write_feature_store_frame
from wc_forecast.storage.model_registry import list_models, register_model
from wc_forecast.storage.prediction_store import (
    read_prediction_ledger,
    write_forecasts_to_prediction_ledger,
)


def test_initialize_database_creates_platform_tables(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"

    initialize_database(database_path)

    assert set(table_names(database_path)) >= {
        "feature_store",
        "injury_suspension_events",
        "market_snapshots",
        "model_registry",
        "player_availability",
        "prediction_ledger",
        "scheduled_prediction_runs",
    }


def test_register_model_writes_model_metadata(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    initialize_database(database_path)

    model_id = register_model(
        model_type="logistic",
        train_cutoff_date="2026-01-01",
        feature_version="v1",
        metrics={"accuracy": 0.59, "log_loss": 0.88},
        artifact_path="outputs/model.pkl",
        notes="test model",
        database_path=database_path,
    )

    models = list_models(database_path)

    assert models[0]["model_id"] == model_id
    assert models[0]["model_type"] == "logistic"


def test_write_feature_store_frame_writes_long_form_features(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    initialize_database(database_path)

    features = pd.DataFrame(
        {
            "date": ["2026-06-20"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "elo_diff_home_minus_away": [25.5],
            "is_neutral": [1.0],
        }
    )

    feature_set_id = write_feature_store_frame(
        features=features,
        entity_columns=["home_team", "away_team"],
        feature_columns=["elo_diff_home_minus_away", "is_neutral"],
        database_path=database_path,
    )

    with sqlite3.connect(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM feature_store WHERE feature_set_id = ?",
            (feature_set_id,),
        ).fetchone()[0]

    assert count == 2


def test_write_forecasts_to_prediction_ledger(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    initialize_database(database_path)

    forecasts = pd.DataFrame(
        {
            "date": ["2026-06-20"],
            "kickoff_at": ["2026-06-20T18:00:00Z"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "predicted_outcome": ["home_win"],
            "predicted_winner": ["Argentina"],
            "model_confidence": [0.61],
            "prob_home_win": [0.61],
            "prob_draw": [0.21],
            "prob_away_win": [0.18],
        }
    )

    inserted = write_forecasts_to_prediction_ledger(
        forecasts,
        model_id=None,
        database_path=database_path,
    )

    ledger = read_prediction_ledger(database_path)

    assert inserted == 1
    assert ledger[0]["home_team"] == "Argentina"
    assert ledger[0]["status"] == "open"


def test_write_external_signal_tables(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    initialize_database(database_path)

    market_count = write_market_snapshots(
        pd.DataFrame(
            {
                "fixture_id": ["2026-06-20:Argentina:Brazil"],
                "snapshot_timestamp": ["2026-06-19T12:00:00Z"],
                "bookmaker": ["synthetic"],
                "home_team": ["Argentina"],
                "away_team": ["Brazil"],
                "home_odds": [2.1],
                "draw_odds": [3.4],
                "away_odds": [3.7],
                "source": ["test"],
            }
        ),
        database_path=database_path,
    )

    availability_count = write_player_availability(
        pd.DataFrame(
            {
                "team": ["Argentina"],
                "player_name": ["Example Player"],
                "as_of": ["2026-06-19T12:00:00Z"],
                "expected_status": ["available"],
                "availability_probability": [0.95],
            }
        ),
        database_path=database_path,
    )

    event_count = write_injury_suspension_events(
        pd.DataFrame(
            {
                "team": ["Brazil"],
                "player_name": ["Example Player"],
                "event_type": ["injury"],
                "status": ["questionable"],
            }
        ),
        database_path=database_path,
    )

    assert market_count == 1
    assert availability_count == 1
    assert event_count == 1


def test_read_latest_prediction_ledger_returns_latest_per_fixture(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    initialize_database(database_path)

    forecasts = pd.DataFrame(
        {
            "date": ["2026-06-20"],
            "kickoff_at": ["2026-06-20T18:00:00Z"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "predicted_outcome": ["home_win"],
            "predicted_winner": ["Argentina"],
            "model_confidence": [0.61],
            "prob_home_win": [0.61],
            "prob_draw": [0.21],
            "prob_away_win": [0.18],
        }
    )

    write_forecasts_to_prediction_ledger(forecasts, database_path=database_path)
    write_forecasts_to_prediction_ledger(forecasts, database_path=database_path)

    from wc_forecast.storage.prediction_store import read_latest_prediction_ledger

    latest_rows = read_latest_prediction_ledger(database_path)

    assert len(latest_rows) == 1
    assert latest_rows[0]["home_team"] == "Argentina"
