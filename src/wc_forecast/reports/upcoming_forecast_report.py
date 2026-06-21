from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_FORECAST_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "predicted_winner",
    "model_confidence",
    "home_elo_rating",
    "away_elo_rating",
}


def build_upcoming_forecast_report(
    forecasts: pd.DataFrame,
    max_rows: int = 10,
    upset_probability_threshold: float = 0.30,
) -> str:
    """Build a Markdown report from upcoming World Cup fixture forecasts."""

    missing_columns = REQUIRED_FORECAST_COLUMNS - set(forecasts.columns)

    if missing_columns:
        raise ValueError(
            f"Forecast file missing required columns: {sorted(missing_columns)}"
        )

    if forecasts.empty:
        raise ValueError("Forecast file contains no rows.")

    data = forecasts.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    data = data.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    high_confidence = data.sort_values(
        "model_confidence",
        ascending=False,
    ).head(max_rows)

    uncertain = data.sort_values(
        "model_confidence",
        ascending=True,
    ).head(max_rows)

    upset_watch = _build_upset_watch(data, upset_probability_threshold).head(max_rows)

    sections = [
        "# Upcoming World Cup Forecast Report",
        "",
        f"Matches forecasted: **{len(data)}**",
        "",
        "## Highest-Confidence Forecasts",
        "",
        _format_forecast_table(high_confidence),
        "",
        "## Most Uncertain Matches",
        "",
        _format_forecast_table(uncertain),
        "",
        "## Potential Upset Watch",
        "",
        _format_upset_table(upset_watch),
        "",
        "## Rating Warnings",
        "",
        _format_rating_warnings(data),
        "",
        "## Notes",
        "",
        (
            "Probabilities represent model-estimated outcome likelihoods, "
            "not guarantees. This report is for analytical and educational use, "
            "not betting advice."
        ),
        "",
    ]

    return "\n".join(sections)


def save_upcoming_forecast_report(
    forecasts_path: str | Path,
    output_path: str | Path,
    max_rows: int = 10,
    upset_probability_threshold: float = 0.30,
) -> str:
    """Load fixture forecasts, build a Markdown report, and save it."""

    forecasts = pd.read_csv(forecasts_path)
    report = build_upcoming_forecast_report(
        forecasts=forecasts,
        max_rows=max_rows,
        upset_probability_threshold=upset_probability_threshold,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report)

    return report


def _build_upset_watch(
    forecasts: pd.DataFrame,
    upset_probability_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in forecasts.to_dict("records"):
        home_rating = float(row["home_elo_rating"])
        away_rating = float(row["away_elo_rating"])
        home_probability = float(row["prob_home_win"])
        away_probability = float(row["prob_away_win"])

        if home_rating < away_rating:
            lower_rated_team = str(row["home_team"])
            higher_rated_team = str(row["away_team"])
            upset_probability = home_probability
            rating_gap = away_rating - home_rating
        else:
            lower_rated_team = str(row["away_team"])
            higher_rated_team = str(row["home_team"])
            upset_probability = away_probability
            rating_gap = home_rating - away_rating

        if upset_probability >= upset_probability_threshold:
            rows.append(
                {
                    "date": row["date"],
                    "match": f'{row["home_team"]} vs {row["away_team"]}',
                    "lower_rated_team": lower_rated_team,
                    "higher_rated_team": higher_rated_team,
                    "rating_gap": rating_gap,
                    "upset_probability": upset_probability,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "match",
                "lower_rated_team",
                "higher_rated_team",
                "rating_gap",
                "upset_probability",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "upset_probability",
        ascending=False,
    )


def _format_forecast_table(forecasts: pd.DataFrame) -> str:
    if forecasts.empty:
        return "_No matches available._"

    rows = [
        "| Date | Match | Pick | Home | Draw | Away | Confidence |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in forecasts.to_dict("records"):
        rows.append(
            "| "
            f"{row['date']} | "
            f"{row['home_team']} vs {row['away_team']} | "
            f"{row['predicted_winner']} | "
            f"{_format_probability(row['prob_home_win'])} | "
            f"{_format_probability(row['prob_draw'])} | "
            f"{_format_probability(row['prob_away_win'])} | "
            f"{_format_probability(row['model_confidence'])} |"
        )

    return "\n".join(rows)


def _format_upset_table(upsets: pd.DataFrame) -> str:
    if upsets.empty:
        return "_No lower-rated teams exceed the upset watch threshold._"

    rows = [
        "| Date | Match | Lower-Rated Team | Higher-Rated Team | Rating Gap | Upset Probability |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in upsets.to_dict("records"):
        rows.append(
            "| "
            f"{row['date']} | "
            f"{row['match']} | "
            f"{row['lower_rated_team']} | "
            f"{row['higher_rated_team']} | "
            f"{float(row['rating_gap']):.1f} | "
            f"{_format_probability(row['upset_probability'])} |"
        )

    return "\n".join(rows)


def _format_rating_warnings(forecasts: pd.DataFrame) -> str:
    if "rating_warning" not in forecasts.columns:
        return "_No rating warning column found._"

    warnings = forecasts[
        forecasts["rating_warning"].fillna("").astype(str).str.strip().ne("")
    ]

    if warnings.empty:
        return "_No rating warnings._"

    rows = [
        "| Date | Match | Warning |",
        "|---|---|---|",
    ]

    for row in warnings.to_dict("records"):
        rows.append(
            "| "
            f"{row['date']} | "
            f"{row['home_team']} vs {row['away_team']} | "
            f"{row['rating_warning']} |"
        )

    return "\n".join(rows)


def _format_probability(value: object) -> str:
    return f"{float(value) * 100:.1f}%"
