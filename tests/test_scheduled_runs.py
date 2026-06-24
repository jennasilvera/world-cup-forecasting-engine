from __future__ import annotations

from wc_forecast.storage.database import initialize_database
from wc_forecast.storage.scheduled_runs import (
    list_scheduled_prediction_runs,
    record_scheduled_prediction_run,
)


def test_record_scheduled_prediction_run(tmp_path) -> None:
    database_path = tmp_path / "engine.sqlite"
    initialize_database(database_path)

    run_id = record_scheduled_prediction_run(
        as_of="2026-06-20T00:00:00Z",
        fixtures_path="data/processed/world_cup_2026_fixtures.csv",
        output_path="outputs/world_cup_2026_upcoming_forecasts.csv",
        status="success",
        forecast_count=4,
        message="scheduled run completed",
        database_path=database_path,
    )

    runs = list_scheduled_prediction_runs(database_path)

    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "success"
    assert runs[0]["forecast_count"] == 4
