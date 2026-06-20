from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.validation.rolling_backtest import (
    DEFAULT_ROLLING_CUTOFF_DATES,
    run_rolling_backtest,
    summarize_rolling_backtest,
)

DEFAULT_HALF_LIFE_DAYS_GRID = [365.0, 730.0, 1095.0, 1460.0, 2190.0, 2920.0]
DEFAULT_LOGISTIC_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]

LOGISTIC_TUNING_COLUMNS = [
    "sample_weight_half_life_days",
    "logistic_c",
    "folds",
    "total_test_rows",
    "mean_accuracy",
    "mean_log_loss",
    "mean_brier",
]


def tune_logistic_model(
    features: pd.DataFrame,
    half_life_days_grid: list[float] | None = None,
    logistic_c_grid: list[float] | None = None,
    cutoff_dates: list[str] | None = None,
    evaluation_window_days: int = 365,
) -> pd.DataFrame:
    """Tune logistic recency half-life and regularization by rolling validation."""

    active_half_lives = half_life_days_grid or DEFAULT_HALF_LIFE_DAYS_GRID
    active_logistic_c = logistic_c_grid or DEFAULT_LOGISTIC_C_GRID
    active_cutoffs = cutoff_dates or DEFAULT_ROLLING_CUTOFF_DATES

    rows: list[dict[str, float]] = []

    for half_life_days in active_half_lives:
        for logistic_c in active_logistic_c:
            rolling_results = run_rolling_backtest(
                features=features,
                cutoff_dates=active_cutoffs,
                model_types=["logistic"],
                evaluation_window_days=evaluation_window_days,
                sample_weight_half_life_days=half_life_days,
                logistic_c=logistic_c,
            )
            summary = summarize_rolling_backtest(rolling_results).iloc[0]

            rows.append(
                {
                    "sample_weight_half_life_days": float(half_life_days),
                    "logistic_c": float(logistic_c),
                    "folds": int(summary["folds"]),
                    "total_test_rows": int(summary["total_test_rows"]),
                    "mean_accuracy": float(summary["mean_accuracy"]),
                    "mean_log_loss": float(summary["mean_log_loss"]),
                    "mean_brier": float(summary["mean_brier"]),
                }
            )

    result = pd.DataFrame(rows, columns=LOGISTIC_TUNING_COLUMNS)
    result = result.sort_values(["mean_log_loss", "mean_brier"]).reset_index(drop=True)

    return result


def save_logistic_tuning(
    features_path: str | Path,
    output_path: str | Path,
    half_life_days_grid: list[float] | None = None,
    logistic_c_grid: list[float] | None = None,
    cutoff_dates: list[str] | None = None,
    evaluation_window_days: int = 365,
) -> pd.DataFrame:
    """Load features, tune logistic model, and save tuning CSV."""

    features = pd.read_csv(features_path)
    result = tune_logistic_model(
        features=features,
        half_life_days_grid=half_life_days_grid,
        logistic_c_grid=logistic_c_grid,
        cutoff_dates=cutoff_dates,
        evaluation_window_days=evaluation_window_days,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)

    return result
