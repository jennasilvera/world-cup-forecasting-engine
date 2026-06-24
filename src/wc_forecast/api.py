from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wc_forecast.forecasting.fixture_forecast import (
    save_upcoming_fixture_forecasts_from_results,
)
from wc_forecast.models.classifier import DEFAULT_RECENCY_HALF_LIFE_DAYS
from wc_forecast.storage.database import DEFAULT_DATABASE_PATH, initialize_database
from wc_forecast.storage.prediction_store import (
    read_latest_prediction_ledger,
    write_forecasts_to_prediction_ledger,
)

DEFAULT_FEATURES_PATH = Path("data/processed/features.csv")
DEFAULT_UPCOMING_FORECAST_RESULTS_PATH = Path("data/processed/results.csv")
DEFAULT_WORLD_CUP_2026_FIXTURES_PATH = Path("data/processed/world_cup_2026_fixtures.csv")
DEFAULT_WORLD_CUP_2026_UPCOMING_FORECASTS_PATH = Path(
    "outputs/world_cup_2026_upcoming_forecasts.csv"
)


class ForecastRunRequest(BaseModel):
    """Request body for running an upcoming-fixture forecast job."""

    fixtures_path: str = Field(
        default=str(DEFAULT_WORLD_CUP_2026_FIXTURES_PATH),
        description="Path to processed fixture CSV.",
    )
    features_path: str = Field(
        default=str(DEFAULT_FEATURES_PATH),
        description="Path to model-ready feature CSV.",
    )
    results_path: str = Field(
        default=str(DEFAULT_UPCOMING_FORECAST_RESULTS_PATH),
        description="Path to processed historical results CSV.",
    )
    output_path: str = Field(
        default=str(DEFAULT_WORLD_CUP_2026_UPCOMING_FORECASTS_PATH),
        description="Path where forecast CSV should be written.",
    )
    train_cutoff_date: str = Field(default="2026-01-01")
    rating_cutoff_date: str | None = Field(default=None)
    as_of: str | None = Field(default=None)
    through_date: str | None = Field(default=None)
    sample_weight_half_life_days: float | None = Field(
        default=DEFAULT_RECENCY_HALF_LIFE_DAYS
    )
    model_type: str = Field(default="logistic")
    logistic_c: float = Field(default=4.0)
    write_ledger: bool = Field(default=True)
    database_path: str = Field(default=str(DEFAULT_DATABASE_PATH))


def create_app() -> FastAPI:
    """Create the forecast API application."""

    app = FastAPI(
        title="World Cup Forecasting Engine API",
        version="0.1.0",
        description="API service for forecast generation and prediction ledger access.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/forecasts/latest")
    def latest_forecasts(
        forecasts_path: str = str(DEFAULT_WORLD_CUP_2026_UPCOMING_FORECASTS_PATH),
    ) -> dict[str, Any]:
        path = Path(forecasts_path)

        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Forecast file not found: {path}",
            )

        forecasts = pd.read_csv(path)

        return {
            "forecast_count": len(forecasts),
            "forecasts": _records(forecasts),
        }

    @app.get("/prediction-ledger/latest")
    def latest_prediction_ledger(
        database_path: str = str(DEFAULT_DATABASE_PATH),
    ) -> dict[str, Any]:
        initialize_database(database_path)
        rows = read_latest_prediction_ledger(database_path)

        return {
            "prediction_count": len(rows),
            "predictions": rows,
        }

    @app.post("/forecasts/run")
    def run_forecast(request: ForecastRunRequest) -> dict[str, Any]:
        try:
            forecasts = save_upcoming_fixture_forecasts_from_results(
                fixtures_path=request.fixtures_path,
                features_path=request.features_path,
                results_path=request.results_path,
                output_path=request.output_path,
                train_cutoff_date=request.train_cutoff_date,
                through_date=request.through_date,
                as_of=request.as_of,
                rating_cutoff_date=request.rating_cutoff_date,
                sample_weight_half_life_days=request.sample_weight_half_life_days,
                model_type=request.model_type,
                logistic_c=request.logistic_c,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        ledger_rows = 0

        if request.write_ledger:
            initialize_database(request.database_path)
            ledger_rows = write_forecasts_to_prediction_ledger(
                forecasts,
                database_path=request.database_path,
            )

        return {
            "status": "success",
            "forecast_count": len(forecasts),
            "ledger_rows_written": ledger_rows,
            "output_path": request.output_path,
            "forecasts": _records(forecasts),
        }

    return app


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe into JSON-safe records."""

    safe = frame.copy()
    safe = safe.where(pd.notna(safe), None)

    return safe.to_dict(orient="records")


app = create_app()
