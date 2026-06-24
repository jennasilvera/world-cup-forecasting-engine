from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from wc_forecast.api import create_app
from wc_forecast.storage.database import initialize_database


def test_api_health() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_latest_forecasts_reads_forecast_csv(tmp_path) -> None:
    forecasts_path = tmp_path / "forecasts.csv"

    pd.DataFrame(
        {
            "date": ["2026-06-20"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "predicted_winner": ["Argentina"],
            "model_confidence": [0.61],
        }
    ).to_csv(forecasts_path, index=False)

    client = TestClient(create_app())

    response = client.get(
        "/forecasts/latest",
        params={"forecasts_path": str(forecasts_path)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast_count"] == 1
    assert payload["forecasts"][0]["home_team"] == "Argentina"


def test_latest_prediction_ledger_returns_rows(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    initialize_database(database_path)

    client = TestClient(create_app())

    response = client.get(
        "/prediction-ledger/latest",
        params={"database_path": str(database_path)},
    )

    assert response.status_code == 200
    assert response.json()["prediction_count"] == 0
