from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTCOME_PROBABILITY_COLUMNS = {
    "home_win": "prob_home_win",
    "draw": "prob_draw",
    "away_win": "prob_away_win",
}


def build_calibration_table(
    predictions: pd.DataFrame,
    probability_column: str,
    outcome_column: str,
    positive_outcome: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a calibration table comparing forecast probability to observed frequency."""

    required_columns = {probability_column, outcome_column}
    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(f"Prediction dataframe missing columns: {sorted(missing_columns)}")

    frame = predictions.copy()
    frame = frame[[probability_column, outcome_column]].dropna()

    if frame.empty:
        raise ValueError("No complete prediction rows available for calibration.")

    frame["forecast_probability"] = frame[probability_column].astype(float).clip(0, 1)
    frame["observed"] = (frame[outcome_column].astype(str) == positive_outcome).astype(int)

    frame["bin"] = pd.cut(
        frame["forecast_probability"],
        bins=pd.interval_range(start=0.0, end=1.0, periods=n_bins, closed="right"),
        include_lowest=True,
    )

    calibration = (
        frame.groupby("bin", observed=False)
        .agg(
            forecast_count=("observed", "size"),
            average_forecast_probability=("forecast_probability", "mean"),
            observed_frequency=("observed", "mean"),
        )
        .reset_index()
    )

    calibration["bin"] = calibration["bin"].astype(str)
    calibration["calibration_error"] = (
        calibration["observed_frequency"]
        - calibration["average_forecast_probability"]
    )

    return calibration


def build_multiclass_calibration_table(
    predictions: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build calibration rows for home/draw/away forecast probabilities."""

    if "realized_outcome" not in predictions.columns:
        raise ValueError("Prediction dataframe missing column: realized_outcome")

    tables = []

    for outcome, probability_column in OUTCOME_PROBABILITY_COLUMNS.items():
        table = build_calibration_table(
            predictions=predictions,
            probability_column=probability_column,
            outcome_column="realized_outcome",
            positive_outcome=outcome,
            n_bins=n_bins,
        )
        table.insert(0, "outcome", outcome)
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def save_calibration_outputs(
    predictions_path: str | Path,
    output_csv_path: str | Path,
    output_plot_path: str | Path,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Save calibration table and plot from a prediction ledger CSV."""

    predictions = pd.read_csv(predictions_path)
    calibration = build_multiclass_calibration_table(predictions, n_bins=n_bins)

    output_csv = Path(output_csv_path)
    output_plot = Path(output_plot_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    calibration.to_csv(output_csv, index=False)
    plot_calibration_table(calibration, output_plot)

    return calibration


def plot_calibration_table(
    calibration: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot calibration curves by outcome."""

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for outcome, group in calibration.groupby("outcome"):
        valid = group.dropna(
            subset=["average_forecast_probability", "observed_frequency"]
        )
        ax.plot(
            valid["average_forecast_probability"],
            valid["observed_frequency"],
            marker="o",
            label=outcome,
        )

    ax.set_title("Forecast Calibration")
    ax.set_xlabel("Average forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
