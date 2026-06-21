from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_AUDIT_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "predicted_outcome",
    "predicted_winner",
    "model_confidence",
    "home_rating_source",
    "away_rating_source",
}


def build_forecast_audit(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Build a compact audit summary for upcoming fixture forecasts."""

    missing_columns = REQUIRED_AUDIT_COLUMNS - set(forecasts.columns)

    if missing_columns:
        raise ValueError(
            f"Forecast file missing required columns: {sorted(missing_columns)}"
        )

    if forecasts.empty:
        raise ValueError("Forecast file contains no rows.")

    data = forecasts.copy()
    confidence = data["model_confidence"].astype(float)

    highest_confidence_row = data.loc[confidence.idxmax()]
    lowest_confidence_row = data.loc[confidence.idxmin()]

    rating_warning_count = 0
    if "rating_warning" in data.columns:
        rating_warning_count = int(
            data["rating_warning"].fillna("").astype(str).str.strip().ne("").sum()
        )

    alias_lookup_count = int(
        (
            data["home_rating_source"].astype(str).eq("alias_lookup")
            | data["away_rating_source"].astype(str).eq("alias_lookup")
        ).sum()
    )

    outcome_distribution = (
        data["predicted_outcome"]
        .value_counts()
        .sort_index()
        .rename_axis("outcome")
        .reset_index(name="count")
    )

    rows: list[dict[str, object]] = [
        {
            "metric": "forecast_count",
            "value": len(data),
        },
        {
            "metric": "average_confidence",
            "value": round(float(confidence.mean()), 6),
        },
        {
            "metric": "highest_confidence_match",
            "value": _format_match(highest_confidence_row),
        },
        {
            "metric": "highest_confidence",
            "value": round(float(highest_confidence_row["model_confidence"]), 6),
        },
        {
            "metric": "lowest_confidence_match",
            "value": _format_match(lowest_confidence_row),
        },
        {
            "metric": "lowest_confidence",
            "value": round(float(lowest_confidence_row["model_confidence"]), 6),
        },
        {
            "metric": "rating_warning_count",
            "value": rating_warning_count,
        },
        {
            "metric": "alias_lookup_count",
            "value": alias_lookup_count,
        },
    ]

    for row in outcome_distribution.to_dict("records"):
        rows.append(
            {
                "metric": f'predicted_{row["outcome"]}_count',
                "value": int(row["count"]),
            }
        )

    return pd.DataFrame(rows)


def save_forecast_audit(
    forecasts_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Load upcoming forecasts, build an audit summary, and save it."""

    forecasts = pd.read_csv(forecasts_path)
    audit = build_forecast_audit(forecasts)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(destination, index=False)

    return audit


def _format_match(row: pd.Series) -> str:
    return (
        f'{row["date"]}: {row["home_team"]} vs {row["away_team"]} '
        f'-> {row["predicted_winner"]}'
    )
