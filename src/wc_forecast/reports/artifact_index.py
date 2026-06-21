from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_ARTIFACT_PATHS = [
    Path("outputs/world_cup_2026_upcoming_forecasts.csv"),
    Path("outputs/world_cup_2026_upcoming_forecast_report.md"),
    Path("outputs/world_cup_2026_upcoming_forecast_audit.csv"),
    Path("outputs/rolling_backtest_metrics.csv"),
    Path("outputs/feature_ablation_results.csv"),
]

DEFAULT_ARTIFACT_LABELS = {
    Path("outputs/world_cup_2026_upcoming_forecasts.csv"): "Upcoming Forecast CSV",
    Path("outputs/world_cup_2026_upcoming_forecast_report.md"): (
        "Upcoming Forecast Report"
    ),
    Path("outputs/world_cup_2026_upcoming_forecast_audit.csv"): (
        "Upcoming Forecast Audit"
    ),
    Path("outputs/rolling_backtest_metrics.csv"): "Rolling Backtest Metrics",
    Path("outputs/feature_ablation_results.csv"): "Feature Ablation Results",
}


def build_artifact_index(paths: list[Path] | None = None) -> pd.DataFrame:
    """Build an index of generated forecast/model artifacts."""

    artifact_paths = paths or DEFAULT_ARTIFACT_PATHS
    rows: list[dict[str, object]] = []

    for path in artifact_paths:
        exists = path.exists()
        stat = path.stat() if exists else None

        rows.append(
            {
                "artifact": DEFAULT_ARTIFACT_LABELS.get(path, path.stem),
                "path": str(path),
                "exists": exists,
                "size_bytes": stat.st_size if stat else None,
                "modified_at": (
                    pd.Timestamp(stat.st_mtime, unit="s").isoformat()
                    if stat
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def save_artifact_index(
    output_path: str | Path,
    paths: list[Path] | None = None,
) -> pd.DataFrame:
    """Save an artifact index CSV."""

    index = build_artifact_index(paths=paths)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(destination, index=False)

    return index
